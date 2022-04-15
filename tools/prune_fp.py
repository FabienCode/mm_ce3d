import argparse
import mmcv
import os

import numpy as np
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.runner.checkpoint import save_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
import torch
import torch.nn as nn


def channel_count_rough(model):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0]  # channels numbers
    return total


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
             'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
             ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function (deprecate), '
             'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
             'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # print(model)

    net_channel_1 = channel_count_rough(model)
    print("The total number of channels in the model before pruning is ", net_channel_1)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    #     outputs = single_gpu_test(model, data_loader)  # , args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)

    obtain_num_parameters = lambda model: sum([param.nelement() for param in model.parameters()])
    origianl_parameters = obtain_num_parameters(model)
    print("\noriginal sum of parameters:", origianl_parameters)
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            # print(dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, **eval_kwargs))

    # ======================== prune model ============================ #
    total_fp0 = 0
    for i, layer in model.named_modules():
        if 'FP_modules.1' in i:
            if isinstance(layer, nn.BatchNorm2d):
                total_fp0 += layer.weight.data.shape[0]

    bn = torch.zeros(total_fp0)
    index = 0
    for i, layer in model.named_modules():
        if 'FP_modules.1' in i:
            if isinstance(layer, nn.BatchNorm2d):
                print(i)
                size = layer.weight.data.shape[0]
                bn[index:(index + size)] = layer.weight.data.abs().clone()
                index += size
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.shape[0]
    #         bn[index:(index + size)] = m.weight.data.abs().clone()
    #         index += size

    y, i = torch.sort(bn)
    thre_index = int(total_fp0 * 0.3)
    thre = y[thre_index]
    pruned = 0
    cfg_pruned = []
    cfg_pruned_mask = []
    k = 0
    for i, m in model.named_modules():
        if 'FP_modules.1' in i:
            if isinstance(m, nn.BatchNorm2d):
                weight_copy = m.weight.data.clone()
                mask = weight_copy.abs().gt(thre).float().cuda()
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                cfg_pruned.append(int(torch.sum(mask)))
                cfg_pruned_mask.append(mask.clone())
                print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(
                    k, mask.shape[0], int(torch.sum(mask)
                                          )))
                k += 1
    pruned_ratio = pruned / total_fp0

    print("Pre-processing Successful")

    # test again
    # if not distributed:
    #     model = MMDataParallel(model, device_ids=[0])
    #     outputs = single_gpu_test(model, data_loader)  # , args.show, args.show_dir)
    # else:
    #     model = MMDistributedDataParallel(
    #         model.cuda(),
    #         device_ids=[torch.cuda.current_device()],
    #         broadcast_buffers=False)
    #     outputs = multi_gpu_test(model, data_loader, args.tmpdir,
    #                              args.gpu_collect)
    print("After Pre-processing mAP\n")
    # print(dataset.evaluate(outputs, show=args.show, out_dir=args.show_dir, **eval_kwargs))

    # Make real prune
    print(cfg)

    # backbone_pruned_sa_channels = ((64, 64, 128),
    #                                (128, 128, 256),
    #                                (128, 128, 256),
    #                                (cfg_pruned[0], cfg_pruned[1], cfg_pruned[2]))
    backbone_pruned_fp_channels = ((256, 256),
                                   (cfg_pruned[0], cfg_pruned[1]))
    # bbox_head_vote_module_cfg_conv_channels = (cfg_pruned[15], cfg_pruned[15])
    # bbox_head_pruned_vote_agg = [cfg_pruned[15], cfg_pruned[16], cfg_pruned[17], cfg_pruned[18]]
    tmp_model_cfg = cfg.model
    # tmp_model_cfg.backbone.sa_channels = backbone_pruned_sa_channels
    tmp_model_cfg.backbone.fp_channels = backbone_pruned_fp_channels
    # tmp_model_cfg.bbox_head.vote_module_cfg.conv_channels = bbox_head_vote_module_cfg_conv_channels
    # tmp_model_cfg.bbox_head.vote_aggregation_cfg.mlp_channels = bbox_head_pruned_vote_agg
    pruned_model = build_model(tmp_model_cfg, test_cfg=cfg.get('test_cfg'))
    if not distributed:
        pruned_model = MMDataParallel(pruned_model, device_ids=[0])
    #   outputs = single_gpu_test(model, data_loader)  # , args.show, args.show_dir)
    else:
        pruned_model = MMDistributedDataParallel(
            pruned_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    layer_id_in_cfg = 0
    start_mask = torch.ones(512)
    end_mask = cfg_pruned_mask[layer_id_in_cfg]
    # i, m in model.named_modules()
    for [(i, m0), m1] in zip(model.named_modules(), pruned_model.modules()):
        if 'FP_modules.1' in i:
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                m1.weight.data = m0.weight.data[idx1].clone()
                m1.bias.data = m0.bias.data[idx1].clone()
                m1.running_mean = m0.running_mean[idx1].clone()
                m1.running_var = m0.running_var[idx1].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_pruned_mask):
                    end_mask = cfg_pruned_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                # if layer_id_in_cfg in {3, 6, 9, 16}:
                #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                #     idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #     print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0] + 3, idx1.shape[0]))
                #     tmp_a = m0.weight.data[:, idx0, :, :].clone()
                #     tmp_b = m0.weight.data[:, -4: -1, :, :].clone()
                #     w = torch.cat((tmp_a, tmp_b), 1)
                #     w = w[idx1, :, :, :].clone()
                #     m1.weight.data = w.clone()
                # elif layer_id_in_cfg == 12:
                #     tmp_start_mask = torch.cat((cfg_pruned_mask[8], cfg_pruned_mask[11]))
                #     idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
                #     idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #     print('In shape: {:d} Out shape:{:d}'.format(cfg_pruned[8] + cfg_pruned[11], idx1.shape[0]))
                #     w = m0.weight.data[:, idx0, :, :].clone()
                #     # tmp_b = m0.weight.data[:, -4: -1, :, :].clone()
                #     # w = torch.cat((tmp_a, tmp_b), 1)
                #     w = w[idx1, :, :, :].clone()
                #     m1.weight.data = w.clone()
                # elif layer_id_in_cfg == 14:
                #     tmp_start_mask = torch.cat((cfg_pruned_mask[5], cfg_pruned_mask[13]))
                #     idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
                #     idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                #     print('In shape: {:d} Out shape:{:d}'.format(cfg_pruned[5] + cfg_pruned[13], idx1.shape[0]))
                #     w = m0.weight.data[:, idx0, :, :].clone()
                #     w = w[idx1, :, :, :].clone()
                #     m1.weight.data = w.clone()
                # else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
                w = m0.weight.data[:, idx0, :, :].clone()
                w = w[idx1, :, :, :].clone()
                m1.weight.data = w.clone()
        # elif isinstance(m0, nn.Conv1d) or isinstance(m0, nn.BatchNorm1d):
        #     m1.weight.data = m0.weight.data.clone()
        # elif 'backbone.SA_modules.3.mlps.0.layer0.conv' in i:
        #     tmp_start_mask = cfg_pruned_mask[2]
        #     idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
        #     tmp_a = m0.weight.data[:, idx0, :, :].clone()
        #     tmp_b = m0.weight.data[:, -4: -1, :, :].clone()
        #     w = torch.cat((tmp_a, tmp_b), 1)
        #     print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))
        #     m1.weight.data = w.clone()

        # ===========
        # elif 'FP_modules.1.mlps.layer0.conv' in i:
        #     tmp_start_mask = torch.cat((cfg_pruned_mask[1].cuda(), torch.ones(256).cuda()))
        #     idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
        #     w = m0.weight.data[:, idx0, :, :].clone()
        #     m1.weight.data = w.clone()
        #     print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))

        # elif 'FP_modules.1.mlps.layer0.conv' in i:
        #     tmp_start_mask = torch.cat((cfg_pruned_mask[2].cuda(), torch.ones(256).cuda()))
        #     idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
        #     w = m0.weight.data[:, idx0, :, :].clone()
        #     m1.weight.data = w.clone()
        #     print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))

        # ====== fp1
        elif 'bbox_head.vote_module.vote_conv.0.conv' in i:
            tmp_start_mask = cfg_pruned_mask[1]
            idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
            w = m0.weight.data[:, idx0, :].clone()
            m1.weight.data = w.clone()
            print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))

        # ====== fp1
        elif 'bbox_head.vote_module.conv_out' in i:
            tmp_start_mask = torch.cat((cfg_pruned_mask[1], torch.ones(3).cuda()))
            idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
            w = m0.weight.data[idx0, :, :].clone()
            m1.weight.data = w.clone()
            m1.bias.data = m0.bias.data[idx0].clone()
            print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))

        # ====== fp1
        elif 'bbox_head.vote_aggregation.mlps.0.layer0.conv' in i:
            tmp_start_mask = torch.cat((cfg_pruned_mask[1], torch.ones(3).cuda()))
            idx0 = np.squeeze(np.argwhere(np.asarray(tmp_start_mask.cpu().numpy())))
            w = m0.weight.data[:, idx0, :].clone()
            m1.weight.data = w.clone()
            print('In shape: {:d} Out shape:{:d}'.format(w.shape[1], w.shape[0]))

        elif isinstance(m0, nn.Conv1d) or isinstance(m0, nn.BatchNorm1d) or isinstance(m0, nn.Conv2d) or isinstance(m0,
                                                                                                                    nn.BatchNorm2d):
            m1.weight.data = m0.weight.data.clone()
    new_parameters = obtain_num_parameters(pruned_model)
    print("\new model sum of parameters:", new_parameters)
    save_checkpoint(pruned_model, 'pruned_fp1_0.3_votenet.pth')
    a = torch.load("pruned_votenet.pth")

    print("hahaha")


if __name__ == '__main__':
    main()
