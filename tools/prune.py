from models.common import *
from models.yolo import *
from models.pruned_common import C3Pruned,SPPFPruned
from copy import deepcopy
from val import *
from terminaltables import AsciiTable
import time
from prune_utils import *
import argparse
import torch
import yaml


def channel_count_rough(model):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            total += m.weight.data.shape[0] # channels numbers
    return total

    # ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='YoloV5 - V6.0')
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/v5.yaml', help='data.yaml path')
    parser.add_argument('--weights', type=str, default='sparse-1-320.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.05, help='channel prune percent')
    parser.add_argument('--imgsz', type=int, default=64, help='inference size (pixels)')
    parser.add_argument('--batch-size', type=int, default=16, help="Total batch size for all gpus.")
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    opt = parser.parse_args()
    print(opt)

    print("=" * 150)
    print("Test before prune:")

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
    train_path = data_dict['train']
    test_path = data_dict['val']
    nc, names = int(data_dict['nc']), data_dict['names']  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(opt.cfg, nc=nc).to(device)

    if opt.weights.endswith('.pt'):  # pytorch format
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    try:
        exclude = []  # exclude keys
        model_items = None
        if isinstance(ckpt['model'], Model):
            model_items = ckpt['model'].float().state_dict().items()
        else:
            model_items = ckpt['model'].items()
        model_items = {k: v for k, v in model_items if k in model.state_dict() and not any(x in k for x in exclude)}
        model.load_state_dict(model_items, strict=True)
        print('Transferred %g/%g items from %s' % (len(model_items), len(model.state_dict()), opt.weights))
    except KeyError as e:
        s = "%s is not compatible with %s. This may be due to model differences or %s may be out of date. " \
            "Please delete or update %s and try again, or use --weights '' to train from scratch." \
            % (opt.weights, opt.cfg, opt.weights, opt.weights)
        raise KeyError(s) from e
    del ckpt

    model.eval()
    net_channel_1 = channel_count_rough(model)
    print("The total number of channels in the model before pruning is ", net_channel_1)
    eval_model = lambda model:run(opt.data, weights=opt.weights, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                device=opt.device, save_json=False, plots=True)

    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)

    origin_nparameters = obtain_num_parameters(model)
    print("original sum of parameters:", origin_nparameters)

    print("=" * 150)
    print("Test after prune:")

    # =========================================== prune model ====================================#
    print("model.module_list:",model.named_children())
    model_list = {}
    ignore_bn_list = []
    for i, layer in model.named_modules():
        if isinstance(layer, Bottleneck):
            if layer.add:  # shortcut前不剪枝nm,.;
                ignore_bn_list.append(i.rsplit(".",2)[0]+".cv1.bn")
                ignore_bn_list.append(i + '.cv1.bn')
                ignore_bn_list.append(i + '.cv2.bn')
        if isinstance(layer, nn.BatchNorm2d):
            if i not in ignore_bn_list:
                model_list[i] = layer
                # print(i, layer)
    model_list = {k: v for k, v in model_list.items() if k not in ignore_bn_list}
    # print("prune module :",model_list.keys())
    prune_conv_list = [layer.replace("bn", "conv") for layer in model_list.keys()]
    # print(prune_conv_list)
    bn_weights = gather_bn_weights(model_list)
    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for bnlayer in model_list.values():
        highest_thre.append(bnlayer.weight.data.abs().max().item())
    # print("highest_thre:",highest_thre)
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn == highest_thre).nonzero()[0, 0].item() / len(bn_weights)
    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.')
    assert opt.percent < percent_limit, f"Prune ratio should less than {percent_limit}, otherwise it may cause error!!!"

    model_copy = deepcopy(model)
    thre_index = int(len(sorted_bn) * opt.percent)
    thre = sorted_bn[thre_index]
    print(f'Gamma value that less than {thre:.4f} are set to zero!')
    print("=" * 94)
    print(f"|\t{'layer name':<25}{'|':<10}{'origin channels':<20}{'|':<10}{'remaining channels':<20}|")
    remain_num = 0
    modelstate = model.state_dict()
    # print(modelstate)

    # ============================================================================== #
    pruned_yaml = {}
    nc = 20
    pruned_yaml["nc"] = 20
    pruned_yaml["depth_multiple"] = 0.33
    pruned_yaml["width_multiple"] = 0.50
    pruned_yaml["anchors"] = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    pruned_yaml["backbone"] = [
            [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
            [-1, 3, C3Pruned, [128]],  # 2
            [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
            [-1, 6, C3Pruned, [256]],  # 4
            [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
            [-1, 9, C3Pruned, [512]],  # 6
            [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
            [-1, 3, C3Pruned, [1024]], # 8
            [-1, 1, SPPFPruned, [1024, [5]]],  # 9
        ]
    pruned_yaml["head"] = [
            [-1, 1, Conv, [512, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 6], 1, Concat, [1]],  # cat backbone P4
            [-1, 3, C3Pruned, [512, False]],  # 13

            [-1, 1, Conv, [256, 1, 1]],
            [-1, 1, nn.Upsample, [None, 2, 'nearest']],
            [[-1, 4], 1, Concat, [1]],  # cat backbone P3
            [-1, 3, C3Pruned, [256, False]],  # 17 (P3/8-small)

            [-1, 1, Conv, [256, 3, 2]],
            [[-1, 14], 1, Concat, [1]],  # cat head P4
            [-1, 3, C3Pruned, [512, False]],  # 20 (P4/16-medium)

            [-1, 1, Conv, [512, 3, 2]],
            [[-1, 10], 1, Concat, [1]],  # cat head P5
            [-1, 3, C3Pruned, [1024, False]],  # 23 (P5/32-large)

            [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
        ]
    # ============================================================================== #
    maskbndict = {}
    for bnname, bnlayer in model.named_modules():
        if isinstance(bnlayer, nn.BatchNorm2d):
            bn_module = bnlayer
            mask = obtain_bn_mask(bn_module, thre)
            if bnname in ignore_bn_list:
                mask = torch.ones(bnlayer.weight.data.size()).cuda()
            maskbndict[bnname] = mask
            # print("mask:",mask)
            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
            bn_module.bias.data.mul_(mask)
            # print("bn_module:", bn_module.bias)
            print(f"|\t{bnname:<25}{'|':<10}{bn_module.weight.data.size()[0]:<20}{'|':<10}{int(mask.sum()):<20}|")
    print("=" * 94)
    # print(maskbndict.keys())

    pruned_model = ModelPruned(maskbndict=maskbndict, cfg=pruned_yaml, ch=3).cuda()
    # Compatibility updates
    for m in pruned_model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    from_to_map = pruned_model.from_to_map
    pruned_model_state = pruned_model.state_dict()
    assert pruned_model_state.keys() == modelstate.keys()
    # print(modelstate.keys())

    layernameL, pruned_layernameL=[], []

    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(),
                                                                      pruned_model.named_modules()):
        layernameL.append(layername)
        pruned_layernameL.append(pruned_layername)
    # print(layernameL)
    # print(pruned_layernameL)

    changed_state = []
    for ((layername, layer), (pruned_layername, pruned_layer)) in zip(model.named_modules(),
                                                                      pruned_model.named_modules()):
        assert layername == pruned_layername
        if isinstance(layer, nn.Conv2d) and not layername.startswith("model.24"):
            convname = layername[:-4] + "bn"
            if convname in from_to_map.keys():
                former = from_to_map[convname]
                if isinstance(former, str):
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
                    w = layer.weight.data[:, in_idx, :, :].clone()
                    w = w[out_idx, :, :, :].clone()
                    if len(w.shape) == 3:
                        w = w.unsqueeze(0)
                    pruned_layer.weight.data = w.clone()
                    changed_state.append(layername + ".weight")
                if isinstance(former, list):
                    orignin = [modelstate[i + ".weight"].shape[0] for i in former]
                    formerin = []
                    for it in range(len(former)):
                        name = former[it]
                        tmp = [i for i in range(maskbndict[name].shape[0]) if maskbndict[name][i] == 1]
                        if it > 0:
                            tmp = [k + sum(orignin[:it]) for k in tmp]
                        formerin.extend(tmp)
                    out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                    w = layer.weight.data[out_idx, :, :, :].clone()
                    pruned_layer.weight.data = w[:, formerin, :, :].clone()
                    changed_state.append(layername + ".weight")
            else:
                out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername[:-4] + "bn"].cpu().numpy())))
                w = layer.weight.data[out_idx, :, :, :].clone()
                assert len(w.shape) == 4
                pruned_layer.weight.data = w.clone()
                changed_state.append(layername + ".weight")

        if isinstance(layer, nn.BatchNorm2d):
            out_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[layername].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[out_idx].clone()
            pruned_layer.bias.data = layer.bias.data[out_idx].clone()
            pruned_layer.running_mean = layer.running_mean[out_idx].clone()
            pruned_layer.running_var = layer.running_var[out_idx].clone()
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")
            changed_state.append(layername + ".running_mean")
            changed_state.append(layername + ".running_var")
            changed_state.append(layername + ".num_batches_tracked")

        if isinstance(layer, nn.Conv2d) and layername.startswith("model.24"):
            former = from_to_map[layername]
            in_idx = np.squeeze(np.argwhere(np.asarray(maskbndict[former].cpu().numpy())))
            pruned_layer.weight.data = layer.weight.data[:, in_idx, :, :]
            pruned_layer.bias.data = layer.bias.data
            changed_state.append(layername + ".weight")
            changed_state.append(layername + ".bias")

    missing = [i for i in pruned_model_state.keys() if i not in changed_state]
    # print("missing:",missing)   missing: ['model.24.anchors']
    pruned_model.eval()
    pruned_model.names = model.names
    # =============================================================================================== #
    torch.save({"model": model}, "orign_model.pt")
    torch.save({"model":pruned_model}, "pruned_model.pt")
    model.cuda().eval()

    eval_pruned_model = lambda model: run(opt.data, weights='pruned_model.pt', batch_size=opt.batch_size, imgsz=opt.imgsz,
                                   conf_thres=.25, iou_thres=.45,device=opt.device, save_json=True, plots=True)
    with torch.no_grad():
        origin_model_metric = eval_pruned_model(model)