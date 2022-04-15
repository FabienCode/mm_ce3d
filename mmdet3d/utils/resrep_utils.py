import time

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import mmcv
from mmcv.cnn import get_model_complexity_info
from mmdet.models.builder import DETECTORS
from mmcv.parallel.utils import is_module_wrapper
from mmcv import Config
import string
import torch.nn.functional as F
from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
from torch.nn import GroupNorm, LayerNorm
from copy import deepcopy
from mmcv.runner.checkpoint import load_state_dict

# from mmcv.runner.checkpoint import save_checkpoint

ORIGINAL_CHANNELS = {'sa_channels': [(64, 64, 128), (128, 128, 256), (128, 128, 256),
                                     (128, 128, 256)],
                     'fp_channels': [(256, 256), (256, 256)]}
ORIGINAL_CHANNELS_list = {'sa_channels': [[64, 64, 128], [128, 128, 256], [128, 128, 256],
                                          [128, 128, 256]],
                          'fp_channels': [[256, 256], [256, 256]]}
FLOPS_TARGET = 0.8
# CHANNELS_TARGET = 2816 * 0.3
NUM_AT_LEAST = 1
THRESH = 1e-5


def get_compactor_mask_dict(model: nn.Module):
    compactor_name_to_mask = {}
    compactor_name_to_kernel_param = {}
    for name, buffer in model.named_buffers():
        if 'compactor.mask' in name:
            compactor_name_to_mask[name.replace('mask', '')] = buffer
            # print(name, buffer.size())
    for name, param in model.named_parameters():
        if 'compactor.pwc.weight' in name:
            compactor_name_to_kernel_param[name.replace('pwc.weight', '')] = param
            # print(name, param.size())
    result = {}
    for name, kernel in compactor_name_to_kernel_param.items():
        mask = compactor_name_to_mask[name]
        num_filters = mask.nelement()
        if kernel.ndimension() == 4:
            if mask.ndimension() == 1:
                broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
                result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
            else:
                assert mask.ndimension() == 4
                result[kernel] = mask
        else:
            assert kernel.ndimension() == 1
            result[kernel] = mask
    # for compactor_para, mask in result.items():
    #     compactor_para.data.requires_grad = True
    return result


def resrep_get_unmasked_deps(origin_deps, model: nn.Module, pacesetter_dict):
    unmasked_deps = np.array(origin_deps)
    for child_module in model.modules():
        if hasattr(child_module, 'conv_idx'):
            layer_ones = child_module.get_num_mask_ones()
            unmasked_deps[child_module.conv_idx] = layer_ones
            # print('cur conv, ', child_module.conv_idx, 'dict is ', pacesetter_dict)
            if pacesetter_dict is not None:
                for follower, pacesetter in pacesetter_dict.items():
                    if pacesetter == child_module.conv_idx:
                        unmasked_deps[follower] = layer_ones
                    # print('cur conv ', child_module.conv_idx, 'follower is ', follower)
    return unmasked_deps


def resrep_mask_model(model: nn.Module):
    original_channels = deepcopy(ORIGINAL_CHANNELS)
    original_flops, original_params = rr_flops(original_channels, 'original_network')
    # print('origin flops ', original_flops)
    cur_deps, metric_dict = resrep_get_deps_and_metric_dict(original_channels, model)
    # print(valve_dict)
    sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)
    # print(sorted_valve_dict)
    # print(sorted_metric_dict)

    # calculate currently network flops
    cur_flops, cur_params = rr_flops(cur_deps, 'cur_network')

    cur_deactivated = get_cur_num_deactivated_filters(cur_deps)

    original_flops = float(original_flops.split()[0])
    cur_flops = float(cur_flops.split()[0])
    # print('now deactivated {} filters'.format(cur_deactivated))
    if cur_flops > FLOPS_TARGET * original_flops:
        next_deactivated_max = cur_deactivated + 2
        # print('next deac max', next_deactivated_max)
    else:
        next_deactivated_max = cur_deactivated

    assert next_deactivated_max > 0
    attempt_deps_list = {'sa_channels': [[64, 64, 128], [128, 128, 256], [128, 128, 256],
                                         [128, 128, 256]],
                         'fp_channels': [[256, 256], [256, 256]]}
    i = 0
    skip_idx = []
    while True:
        attempt_deps = {'sa_channels': [tuple(attempt_deps_list['sa_channels'][0]),
                                        tuple(attempt_deps_list['sa_channels'][1]),
                                        tuple(attempt_deps_list['sa_channels'][2]),
                                        tuple(attempt_deps_list['sa_channels'][3])],
                        'fp_channels': [(256, 256), (256, 256)]}
        # attempt_flops, attempt_params = rr_flops(attempt_deps, 'attempt_network')
        # attempt_flops = float(attempt_flops.split()[0])
        # print('attempt flops ', attempt_flops)
        # if attempt_flops <= FLOPS_TARGET * original_flops:
        #     break

        # pruning ======================
        attempt_layer_filter = sorted_metric_dict[i]
        if attempt_deps_list['sa_channels'][int(attempt_layer_filter[0] / 2)][attempt_layer_filter[0] % 2 + 1] \
                <= NUM_AT_LEAST:
            skip_idx.append(i)
            i += 1
            continue
        attempt_deps_list['sa_channels'][int(attempt_layer_filter[0] / 2)][attempt_layer_filter[0] % 2 + 1] -= 1
        i += 1
        if i >= next_deactivated_max:
            break

    layer_masked_out_filters = defaultdict(list)  # layer_idx : [zeros]
    for k in range(i):
        if k not in skip_idx:
            layer_masked_out_filters[sorted_metric_dict[k][0]].append(sorted_metric_dict[k][1])

    set_model_masks(model, layer_masked_out_filters)
    attempt_flops, attempt_params = rr_flops(attempt_deps, 'attempt_network')
    attempt_flops = float(attempt_flops.split()[0])
    return attempt_deps_list, attempt_flops


def get_cur_num_deactivated_filters(cur_deps):
    # assert len(origin_deps) == len(cur_deps)
    # diff = origin_deps - cur_deps
    # assert np.sum(diff < 0) == 0
    origin_deps = deepcopy(ORIGINAL_CHANNELS)

    result = 0
    for i in range(len(origin_deps['sa_channels'])):
        for j in range(len(origin_deps['sa_channels'][i])):
            result += origin_deps['sa_channels'][i][j] - cur_deps['sa_channels'][i][j]

    for i in range(len(origin_deps['fp_channels'])):
        for j in range(len(origin_deps['fp_channels'][i])):
            result += origin_deps['fp_channels'][i][j] - cur_deps['fp_channels'][i][j]

    return result


def set_model_masks(model, layer_masked_out_filters):
    i = 0
    for child_module in model.modules():
        if hasattr(child_module, 'mask'):
            if i in layer_masked_out_filters:
                child_module.set_mask(layer_masked_out_filters[i])
            i += 1


def resrep_get_deps_and_metric_dict(origin_channels, model: nn.Module):
    new_deps = deepcopy(origin_channels)
    layer_ones, metric_dict = resrep_get_layer_mask_ones_and_metric_dict(model)
    new_deps['sa_channels'] = [(64, layer_ones[0], layer_ones[1]), (128, layer_ones[2], layer_ones[3]),
                               (128, layer_ones[4], layer_ones[5]),
                               (128, layer_ones[6], layer_ones[7])]
    return new_deps, metric_dict


#   pacesetter is not included here
def resrep_get_layer_mask_ones_and_metric_dict(model: nn.Module):
    layer_mask_ones = {}
    layer_metric_dict = {}
    report_deps = []
    # model = model.modules()
    i = 0
    for child_module in model.modules():
        # print(child_module)
        if hasattr(child_module, 'mask'):

            layer_mask_ones[i] = child_module.get_num_mask_ones()
            metric_vector = child_module.get_metric_vector()
            # print('cur conv idx', child_module.conv_idx)
            # if len(metric_vector <= 512):
            #     print(metric_vector)#TODO
            for j in range(len(metric_vector)):
                layer_metric_dict[(i, j)] = metric_vector[j]
            report_deps.append(layer_mask_ones[i])
            i += 1
    # print('now active deps: ', report_deps)
    return layer_mask_ones, layer_metric_dict


def rr_flops(channel, network_name, print_log=False):
    cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
    cfg_model = cfg.model
    cfg_model.backbone.sa_channels = channel['sa_channels']
    cfg_model.backbone.fp_channels = channel['fp_channels']
    model = DETECTORS.build(
        cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                model.__class__.__name__))
    input_shape = (40000, 4)
    flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=print_log)
    # split_line = '=' * 30
    # print(f'{split_line}\nInput shape: {input_shape}\n'
    #       f'{network_name}Flops is: {flops}\nParams: {params}\n{split_line}')
    # print('!!!Please be cautious if you use the results in papers. '
    #       'You may need to check if all ops are supported and verify that the '
    #       'flops computation is correct.')
    return flops, params


def compactor_convert(model):
    thresh = THRESH
    compactor_mats = {}
    compactor_list = [2, 4, 7, 9, 12, 14, 17, 19]
    compactor_list_to_cfg = {2: 0, 4: 1, 7: 2, 9: 3, 12: 4, 14: 5, 17: 6, 19: 7}

    i = 0
    for submodule in model.modules():
        if hasattr(submodule, 'mask'):
            compactor_mats[compactor_list[i]] = submodule.pwc.weight.detach().cpu().numpy()
            i += 1

    pruned_deps = deepcopy(ORIGINAL_CHANNELS_list)

    cur_conv_idx = -1
    pop_name_set = set()

    kernel_name_list = []
    fully_kernel_name_list = []
    save_dict = {}
    for k, v in model.state_dict().items():
        v = v.detach().cpu().numpy()
        if v.ndim in [2, 4] and 'compactor.pwc' not in k:
            kernel_name_list.append(k)
            fully_kernel_name_list.append(k)
        elif v.ndim in [2, 4]:
            fully_kernel_name_list.append(k)
        save_dict[k] = v

    for conv_id, kernel_name in enumerate(fully_kernel_name_list):
        if conv_id in compactor_list:
            continue

        kernel_value = save_dict[kernel_name]
        if kernel_value.ndim == 2:
            continue
        fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)
        if_compactor = conv_id + 1
        fold_direct = if_compactor in compactor_mats
        if fold_direct:
            fm = compactor_mats[if_compactor]
            fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)
            pruned_deps['sa_channels'][int(compactor_list_to_cfg[if_compactor] / 2)][
                compactor_list_to_cfg[if_compactor] % 2 + 1] -= len(pruned_ids)
            if len(pruned_ids) > 0:
                fo_kernel_name = fully_kernel_name_list[conv_id + 2]
                fo_value = save_dict[fo_kernel_name]
                if fo_value.ndim == 4:
                    fo_value = np.delete(fo_value, pruned_ids, axis=1)
                else:
                    fc_idx_to_delete = []
                    num_filters = kernel_value.shape[0]
                    fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
                    print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                    base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
                    for i in pruned_ids:
                        fc_idx_to_delete.append(base + i)
                    if len(fc_idx_to_delete) > 0:
                        fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
                save_dict[fo_kernel_name] = fo_value

        save_dict[kernel_name] = fused_k
        save_dict[kernel_name.replace('.weight', '.bias')] = fused_b
    for name in pop_name_set:
        save_dict.pop(name)

    final_dict = {k.replace('module.', ''): v for k, v in save_dict.items() if
                  'num_batches' not in k and 'compactor' not in k}
    cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
    cfg_model = cfg.model
    cfg_model.backbone.sa_channels = pruned_deps['sa_channels']
    cfg_model.backbone.fp_channels = pruned_deps['fp_channels']
    new_model = DETECTORS.build(
        cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))

    if torch.cuda.is_available():
        new_model.cuda()
    new_model.eval()

    if hasattr(model, 'forward_dummy'):
        new_model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                new_model.__class__.__name__))

    # fuse_compactor(model, kernel_name, fused_k, fused_b)
    new_model = fuse_compactor(new_model, final_dict)

    return new_model

def fuse_compactor(module, save_dict):
    # new_model_kernel_name = []
    # new_model_save_dict = {}
    #
    # for k, v in module.state_dict().items():
    #     v = v.detach().cpu().numpy()
    #     if v.ndim in [2, 4]:
    #         new_model_kernel_name.append(k)
    #     new_model_save_dict[k] = v


    last_conv = None
    last_conv_name = None
    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            module._modules[last_conv_name].weight = save_dict[last_conv_name.weight]
            module._modules[last_conv_name].bias = save_dict[last_conv_name.bias]
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, nn.Conv2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_compactor(child, save_dict)
    return module



def fuse_conv_bn(save_dict, pop_name_set, kernel_name):
    mean_name = kernel_name.replace('.conv.weight', '.bn.running_mean')
    var_name = kernel_name.replace('.conv.weight', '.bn.running_var')
    gamma_name = kernel_name.replace('.conv.weight', '.bn.weight')
    beta_name = kernel_name.replace('.conv.weight', '.bn.bias')
    pop_name_set.add(mean_name)
    pop_name_set.add(var_name)
    pop_name_set.add(gamma_name)
    pop_name_set.add(beta_name)
    mean = save_dict[mean_name]
    var = save_dict[var_name]
    gamma = save_dict[gamma_name]
    beta = save_dict[beta_name]
    kernel_value = save_dict[kernel_name]
    # print('kernel name', kernel_name)
    # print('kernel, mean, var, gamma, beta', kernel_value.shape, mean.shape, var.shape, gamma.shape, beta.shape)
    return _fuse_kernel(kernel_value, gamma, var, eps=1e-5), _fuse_bias(mean, var, gamma, beta, eps=1e-5)


def _fuse_kernel(kernel, gamma, running_var, eps):
    print('fusing: kernel shape', kernel.shape)
    std = np.sqrt(running_var + eps)
    t = gamma / std
    t = np.reshape(t, (-1, 1, 1, 1))
    print('fusing: t', t.shape)
    t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
    return kernel * t


def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
    if bias is None:
        return beta - running_mean * gamma / np.sqrt(running_var + eps)
    else:
        return beta + (bias - running_mean) * gamma / np.sqrt(running_var + eps)


def fold_conv(fused_k, fused_b, thresh, compactor_mat):
    metric_vec = np.sqrt(np.sum(compactor_mat ** 2, axis=(1, 2, 3)))
    filter_ids_below_thresh = np.where(metric_vec < thresh)[0]

    if len(filter_ids_below_thresh) == len(metric_vec):
        sortd_ids = np.argsort(metric_vec)
        filter_ids_below_thresh = sortd_ids[:-1]  # TODO preserve at least one filter

    if len(filter_ids_below_thresh) > 0:
        compactor_mat = np.delete(compactor_mat, filter_ids_below_thresh, axis=0)

    kernel = F.conv2d(torch.from_numpy(fused_k).permute(1, 0, 2, 3), torch.from_numpy(compactor_mat),
                      padding=(0, 0)).permute(1, 0, 2, 3)
    Dprime = compactor_mat.shape[0]
    bias = np.zeros(Dprime)
    for i in range(Dprime):
        bias[i] = fused_b.dot(compactor_mat[i, :, 0, 0])

    if type(bias) is not np.ndarray:
        bias = np.array([bias])

    return kernel, bias, filter_ids_below_thresh


# def add_params(params, module, prefix='', is_dcn_module=None):
#     is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
#     is_dwconv = (isinstance(module, torch.nn.Conv2d)
#                  and module.in_channels == module.groups)
#     for name, param in module.named_parameters(recurse=False):
#         if
#         params_group = {'params': [param]}
#         if not param.requires_grad:
#             params.append(params_group)
#             continue
#
#
# def compactor_mm(model):
#     if hasattr(model, 'module'):
#         model = model.module
#
#     compactor_paras = []

def _fuse_conv_bn(conv, bn):
    """Fuse conv and bn into one module.

    Args:
        conv (nn.Module): Conv to be fused.
        bn (nn.Module): BN to be fused.

    Returns:
        nn.Module: Fused module.
    """
    conv_w = conv.weight
    conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
        bn.running_mean)

    factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    conv.weight = nn.Parameter(conv_w *
                               factor.reshape([conv.out_channels, 1, 1, 1]))
    conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
    return conv
