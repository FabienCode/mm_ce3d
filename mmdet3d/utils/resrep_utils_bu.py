# import numpy as np
# import torch
# import torch.nn as nn
# from collections import defaultdict
# from mmcv.cnn import get_model_complexity_info
# from mmdet.models.builder import DETECTORS
# from mmcv import Config
# import string
# import torch.nn.functional as F
# from mmcv.utils import _BatchNorm, _InstanceNorm, build_from_cfg, is_list_of
# from torch.nn import GroupNorm, LayerNorm
# from copy import deepcopy
#
# ORIGINAL_CHANNELS = {'sa_channels': [(64, 64, 128), (128, 128, 256), (128, 128, 256),
#                                      (128, 128, 256)],
#                      'fp_channels': [(256, 256), (256, 256)]}
# ORIGINAL_CHANNELS_list = {'sa_channels': [[64, 64, 128], [128, 128, 256], [128, 128, 256],
#                                           [128, 128, 256]],
#                           'fp_channels': [[256, 256], [256, 256]]}
# FLOPS_TARGET = 0.41
# NUM_AT_LEAST = 1
# THRESH = 1e-4
#
#
# def get_compactor_mask_dict(model: nn.Module):
#     compactor_name_to_mask = {}
#     compactor_name_to_kernel_param = {}
#     for name, buffer in model.named_buffers():
#         if 'compactor.mask' in name:
#             compactor_name_to_mask[name.replace('mask', '')] = buffer
#             # print(name, buffer.size())
#     for name, param in model.named_parameters():
#         if 'compactor.pwc.weight' in name:
#             compactor_name_to_kernel_param[name.replace('pwc.weight', '')] = param
#             # print(name, param.size())
#     result = {}
#     for name, kernel in compactor_name_to_kernel_param.items():
#         mask = compactor_name_to_mask[name]
#         num_filters = mask.nelement()
#         if kernel.ndimension() == 4:
#             if mask.ndimension() == 1:
#                 broadcast_mask = mask.reshape(-1, 1).repeat(1, num_filters)
#                 result[kernel] = broadcast_mask.reshape(num_filters, num_filters, 1, 1)
#             else:
#                 assert mask.ndimension() == 4
#                 result[kernel] = mask
#         else:
#             assert kernel.ndimension() == 1
#             result[kernel] = mask
#     # for compactor_para, mask in result.items():
#     #     compactor_para.data.requires_grad = True
#     return result
#
#
# def resrep_get_unmasked_deps(origin_deps, model: nn.Module, pacesetter_dict):
#     unmasked_deps = np.array(origin_deps)
#     for child_module in model.modules():
#         if hasattr(child_module, 'conv_idx'):
#             layer_ones = child_module.get_num_mask_ones()
#             unmasked_deps[child_module.conv_idx] = layer_ones
#             # print('cur conv, ', child_module.conv_idx, 'dict is ', pacesetter_dict)
#             if pacesetter_dict is not None:
#                 for follower, pacesetter in pacesetter_dict.items():
#                     if pacesetter == child_module.conv_idx:
#                         unmasked_deps[follower] = layer_ones
#                     # print('cur conv ', child_module.conv_idx, 'follower is ', follower)
#     return unmasked_deps
#
#
# def resrep_mask_model(model: nn.Module):
#     original_channels = deepcopy(ORIGINAL_CHANNELS)
#     original_flops, original_params = rr_flops(original_channels, 'original_network')
#     # print('origin flops ', original_flops)
#     cur_deps, metric_dict = resrep_get_deps_and_metric_dict(original_channels, model)
#     # print(valve_dict)
#     sorted_metric_dict = sorted(metric_dict, key=metric_dict.get)
#     # print(sorted_valve_dict)
#     # print(sorted_metric_dict)
#
#     # calculate currently network flops
#     cur_flops, cur_params = rr_flops(cur_deps, 'cur_network')
#
#     cur_deactivated = get_cur_num_deactivated_filters(cur_deps)
#
#     original_flops = float(original_flops.split()[0])
#     cur_flops = float(cur_flops.split()[0])
#     # print('now deactivated {} filters'.format(cur_deactivated))
#     if cur_flops > FLOPS_TARGET * original_flops:
#         next_deactivated_max = cur_deactivated + 8
#         # print('next deac max', next_deactivated_max)
#     else:
#         next_deactivated_max = cur_deactivated
#
#     assert next_deactivated_max > 0
#
#     # # ============= lamp BEGIN
#     #
#     # compactor_score = get_flatted_scores(model)
#     # concat_scores = torch.cat(compactor_score, dim=0)
#     # topks, _ = torch.topk(concat_scores, next_deactivated_max)
#     # threshold = topks[-1]
#     #
#     # final_survs = [torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in
#     #                compactor_score]
#     # # ============ lamp END
#
#     attempt_deps_list = {'sa_channels': [[64, 64, 128], [128, 128, 256], [128, 128, 256],
#                                          [128, 128, 256]],
#                          'fp_channels': [[256, 256], [256, 256]]}
#     i = 0
#     skip_idx = []
#     while True:
#         attempt_deps = {'sa_channels': [tuple(attempt_deps_list['sa_channels'][0]),
#                                         tuple(attempt_deps_list['sa_channels'][1]),
#                                         tuple(attempt_deps_list['sa_channels'][2]),
#                                         tuple(attempt_deps_list['sa_channels'][3])],
#                         'fp_channels': [tuple(attempt_deps_list['fp_channels'][0]),
#                                         tuple(attempt_deps_list['fp_channels'][1])]}
#         # attempt_flops, attempt_params = rr_flops(attempt_deps, 'attempt_network')
#         # attempt_flops = float(attempt_flops.split()[0])
#         # # print('attempt flops ', attempt_flops)
#         # if attempt_flops <= FLOPS_TARGET * original_flops:
#         #     break
#
#         # pruning ======================
#         attempt_layer_filter = sorted_metric_dict[i]
#         if attempt_layer_filter[0] <= 11:
#             if attempt_deps_list['sa_channels'][int(attempt_layer_filter[0] / 3)][attempt_layer_filter[0] % 3] \
#                     <= NUM_AT_LEAST:
#                 skip_idx.append(i)
#                 i += 1
#                 continue
#             attempt_deps_list['sa_channels'][int(attempt_layer_filter[0] / 3)][attempt_layer_filter[0] % 3] -= 1
#             i += 1
#             if i >= next_deactivated_max:
#                 break
#         else:
#             if attempt_deps_list['fp_channels'][int((attempt_layer_filter[0] - 12) / 2)][int((attempt_layer_filter[0] - 12) % 2)] <= NUM_AT_LEAST:
#                 skip_idx.append(i)
#                 i += 1
#                 continue
#             attempt_deps_list['fp_channels'][int((attempt_layer_filter[0] - 12) / 2)][int((attempt_layer_filter[0] - 12) % 2)] -= 1
#             i += 1
#             if i >= next_deactivated_max:
#                 break
#
#     layer_masked_out_filters = defaultdict(list)  # layer_idx : [zeros]
#     for k in range(i):
#         if k not in skip_idx:
#             layer_masked_out_filters[sorted_metric_dict[k][0]].append(sorted_metric_dict[k][1])
#     attempt_flops, attempt_params = rr_flops(attempt_deps, 'attempt_network')
#     attempt_flops = float(attempt_flops.split()[0])
#
#     set_model_masks(model, layer_masked_out_filters)
#     return attempt_deps_list, attempt_flops, attempt_params
#
#
# def get_cur_num_deactivated_filters(cur_deps):
#     # assert len(origin_deps) == len(cur_deps)
#     # diff = origin_deps - cur_deps
#     # assert np.sum(diff < 0) == 0
#     origin_deps = deepcopy(ORIGINAL_CHANNELS)
#
#     result = 0
#     for i in range(len(origin_deps['sa_channels'])):
#         for j in range(len(origin_deps['sa_channels'][i])):
#             result += origin_deps['sa_channels'][i][j] - cur_deps['sa_channels'][i][j]
#
#     for i in range(len(origin_deps['fp_channels'])):
#         for j in range(len(origin_deps['fp_channels'][i])):
#             result += origin_deps['fp_channels'][i][j] - cur_deps['fp_channels'][i][j]
#
#     return result
#
#
# def sum_list(a):
#     sum = 0
#     for i in range(len(a)):
#         sum += a[i]
#
#     return sum
#
#
# def set_model_masks(model, layer_masked_out_filters):
#     i = 0
#     for child_module in model.modules():
#         if hasattr(child_module, 'mask'):
#             if i in layer_masked_out_filters:
#                 child_module.set_mask(layer_masked_out_filters[i])
#             i += 1
#
#
# def get_deps_if_prune_low_metric(model):
#     threshold = THRESH
#     new_deps = deepcopy(ORIGINAL_CHANNELS)
#     layer_ones = {}
#     pruned_ids = []
#     i = 0
#     for child_module in model.modules():
#         if hasattr(child_module, 'mask'):
#             metric_vector = child_module.get_metric_vector()
#
#             tmp_pruned = np.where(metric_vector < THRESH)[0]
#             if len(tmp_pruned) == len(metric_vector):
#                 sorted_ids = np.argsort(metric_vector)
#                 tmp_pruned = sorted_ids[:-1]
#             pruned_ids.append(tmp_pruned)
#
#             num_filters_under_thres = np.sum(metric_vector >= threshold)
#             layer_ones[i] = num_filters_under_thres
#             i += 1
#     new_deps['sa_channels'] = [(layer_ones[0], layer_ones[1], layer_ones[2]),
#                                (layer_ones[3], layer_ones[4], layer_ones[5]),
#                                (layer_ones[6], layer_ones[7], layer_ones[8]),
#                                (layer_ones[9], layer_ones[10], layer_ones[11])]
#     new_deps['fp_channels'] = [(layer_ones[12], layer_ones[13]), (layer_ones[14], 256)]
#     return new_deps, pruned_ids
#
#
# def resrep_get_deps_and_metric_dict(origin_channels, model: nn.Module):
#     new_deps = deepcopy(origin_channels)
#     layer_ones, metric_dict = resrep_get_layer_mask_ones_and_metric_dict(model, threshold=THRESH)
#     new_deps['sa_channels'] = [(layer_ones[0], layer_ones[1], layer_ones[2]),
#                                (layer_ones[3], layer_ones[4], layer_ones[5]),
#                                (layer_ones[6], layer_ones[7], layer_ones[8]),
#                                (layer_ones[9], layer_ones[10], layer_ones[11])]
#     new_deps['fp_channels'] = [(layer_ones[12], layer_ones[13]), (layer_ones[14], 256)]
#     return new_deps, metric_dict
#
#
# #   pacesetter is not included here
# def resrep_get_layer_mask_ones_and_metric_dict(model: nn.Module, threshold):
#     normal_factor = 1e-4
#     layer_mask_ones = {}
#     layer_metric_dict = {}
#     report_deps = []
#     # model = model.modules()
#
#     # ============ dis points features
#     dis_channels_norm = []
#
#     # $$$$$$$$$$$  cof operation ^^^^^^^^
#     cof_channels_norm = []
#     for child_module in model.modules():
#         if hasattr(child_module, 'cof_channels_norm'):
#             tmp_module_channel_norm = child_module.cof_channels_norm
#             for i in range(len(tmp_module_channel_norm)):
#                 cof_channels_norm.append(tmp_module_channel_norm[i])
#         if hasattr(child_module, 'dis_points_norm'):
#             tmp_module_channel_norm = child_module.dis_points_norm
#             for i in range(len(tmp_module_channel_norm)):
#                 dis_channels_norm.append(tmp_module_channel_norm[i])
#     # dis_channels_norm.pop()
#     if len(cof_channels_norm) != 0:
#         cof_channels_norm.pop()
#     # ======= sa cof_norm
#     i = 0
#     for child_module in model.modules():
#         # print(child_module)
#         if hasattr(child_module, 'mask'):
#             # metric_vector = child_module.get_metric_vector()
#             # num_filters_under_thres = np.sum(metric_vector < threshold)
#             layer_mask_ones[i] = child_module.get_num_mask_ones()
#             metric_vector = child_module.get_metric_vector()
#             if len(dis_channels_norm) != 0:
#                 if i <= 11:
#                     metric_vector += dis_channels_norm[
#                                          i].detach().cpu().numpy() * normal_factor  # after cof_attention metric
#             if len(cof_channels_norm) != 0:
#                 metric_vector += cof_channels_norm[
#                                      i].detach().cpu().numpy() * normal_factor  # after cof_attention metric
#             for j in range(len(metric_vector)):
#                 layer_metric_dict[(i, j)] = metric_vector[j]
#             report_deps.append(layer_mask_ones[i])
#             i += 1
#     # print('now active deps: ', report_deps)
#     return layer_mask_ones, layer_metric_dict
#
#
# def get_flatted_scores(model):
#     compactor_weight = []
#     for child_module in model.modules():
#         if hasattr(child_module, 'mask'):
#             new_compactor_value = child_module.pwc.weight.data.detach().cpu().numpy()
#             zeros_indices = np.where(child_module.mask.cpu().numpy() == 0)
#             new_compactor_value[np.array(zeros_indices), :, :, :] = 0.0
#             compactor_weight.append(torch.from_numpy(new_compactor_value).cuda().type(torch.cuda.FloatTensor))
#             # compactor_weight.append(torch.mul(child_module.pwc.weight, child_module.mask))
#             # compactor_weight.append(child_module.pwc.weight)
#
#     flattend_scores = [normalize_scores(w ** 2).view(-1) for w in compactor_weight]
#
#     return flattend_scores
#
#
# def rr_flops(channel, network_name, print_log=False):
#     cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
#     # cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512.py')
#     cfg_model = cfg.model
#     cfg_model.backbone.sa_channels = channel['sa_channels']
#     cfg_model.backbone.fp_channels = channel['fp_channels']
#     model = DETECTORS.build(
#         cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
#     if torch.cuda.is_available():
#         model.cuda()
#     model.eval()
#
#     if hasattr(model, 'forward_dummy'):
#         model.forward = model.forward_dummy
#     else:
#         raise NotImplementedError(
#             'FLOPs counter is currently not supported for {}'.format(
#                 model.__class__.__name__))
#     input_shape = (40000, 4)
#     # input_shape = (40000, 3) # for gp3d input
#     flops, params = get_model_complexity_info(model, input_shape, print_per_layer_stat=print_log)
#     # split_line = '=' * 30
#     # print(f'{split_line}\nInput shape: {input_shape}\n'
#     #       f'{network_name}Flops is: {flops}\nParams: {params}\n{split_line}')
#     # print('!!!Please be cautious if you use the results in papers. '
#     #       'You may need to check if all ops are supported and verify that the '
#     #       'flops computation is correct.')
#     return flops, params
#
#
# def normalize_scores(scores):
#     """
#     Normalizing scheme for LAMP.
#     """
#     # sort scores in an ascending order
#     sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
#     # compute cumulative sum
#     scores_cumsum_temp = sorted_scores.cumsum(dim=0)
#     scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
#     scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp) - 1]
#     # normalize by cumulative sum
#     sorted_scores /= (scores.sum() - scores_cumsum)
#     # tidy up and output
#     new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
#     new_scores[sorted_idx] = sorted_scores
#
#     return new_scores.view(scores.shape)
#
#
# def compactor_convert(model, checkpoint):
#     thresh = THRESH
#     compactor_mats = {}
#     compactor_list = [2, 4, 7, 9, 12, 14, 17, 19, 22]
#     compactor_list_to_cfg = {2: 0, 4: 1, 7: 2, 9: 3, 12: 4, 14: 5, 17: 6, 19: 7, 22: 8}
#
#     i = 0
#     for submodule in model.modules():
#         if hasattr(submodule, 'mask'):
#             compactor_mats[compactor_list[i]] = submodule.pwc.weight.detach().cpu().numpy()
#             i += 1
#
#     pruned_deps = deepcopy(ORIGINAL_CHANNELS_list)
#     pop_name_set = set()
#
#     kernel_name_list = []
#     fully_kernel_name_list = []
#     save_dict = {}
#     for k, v in model.state_dict().items():
#         v = v.detach().cpu().numpy()
#         if v.ndim in [2, 4] and 'compactor.pwc' not in k:
#             kernel_name_list.append(k)
#             fully_kernel_name_list.append(k)
#         elif v.ndim in [2, 4]:
#             fully_kernel_name_list.append(k)
#         save_dict[k] = v
#
#     for conv_id, kernel_name in enumerate(fully_kernel_name_list):
#         if conv_id in compactor_list:
#             continue
#
#         kernel_value = save_dict[kernel_name]
#         if kernel_value.ndim == 2:
#             continue
#         fused_k, fused_b = fuse_conv_bn(save_dict, pop_name_set, kernel_name)
#         # fused_k, fused_b = _fuse_conv_bn(model, kernel_name)
#         if_compactor = conv_id + 1
#         fold_direct = if_compactor in compactor_mats
#         if fold_direct:
#             fm = compactor_mats[if_compactor]
#             fused_k, fused_b, pruned_ids = fold_conv(fused_k, fused_b, thresh, fm)
#             if conv_id <= 20:
#                 pruned_deps['sa_channels'][int(compactor_list_to_cfg[if_compactor] / 2)][
#                     compactor_list_to_cfg[if_compactor] % 2 + 1] -= len(pruned_ids)
#             else:
#                 pruned_deps['fp_channels'][0][1] -= len(pruned_ids)
#             if len(pruned_ids) > 0:
#                 if conv_id == 8:
#                     fo_kernel_name = fully_kernel_name_list[23]
#                     fo_value = save_dict[fo_kernel_name]
#                     fo_value = np.delete(fo_value, pruned_ids, axis=1)
#                     save_dict[fo_kernel_name] = fo_value
#
#                 if conv_id == 13:
#                     fo_kernel_name = fully_kernel_name_list[20]
#                     fo_value = save_dict[fo_kernel_name]
#                     fo_value = np.delete(fo_value, pruned_ids, axis=1)
#                     save_dict[fo_kernel_name] = fo_value
#                 if conv_id == 18:
#                     pruned_ids += pruned_deps['sa_channels'][2][2]
#                 fo_kernel_name = fully_kernel_name_list[conv_id + 2]
#                 fo_value = save_dict[fo_kernel_name]
#                 if fo_value.ndim == 4:
#                     fo_value = np.delete(fo_value, pruned_ids, axis=1)
#                 else:
#                     fc_idx_to_delete = []
#                     num_filters = kernel_value.shape[0]
#                     fc_neurons_per_conv_kernel = fo_value.shape[1] // num_filters
#                     print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
#                     base = np.arange(0, fc_neurons_per_conv_kernel * num_filters, num_filters)
#                     for i in pruned_ids:
#                         fc_idx_to_delete.append(base + i)
#                     if len(fc_idx_to_delete) > 0:
#                         fo_value = np.delete(fo_value, np.concatenate(fc_idx_to_delete, axis=0), axis=1)
#                 save_dict[fo_kernel_name] = fo_value
#
#         save_dict[kernel_name] = fused_k
#         save_dict[kernel_name.replace('.weight', '.bias')] = fused_b
#     for name in pop_name_set:
#         save_dict.pop(name)
#
#     final_dict = {k.replace('module.', ''): v for k, v in save_dict.items() if
#                   'num_batches' not in k and 'compactor' not in k}
#     cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
#     cfg_model = cfg.model
#     cfg_model.backbone.sa_channels = pruned_deps['sa_channels']
#     cfg_model.backbone.fp_channels = pruned_deps['fp_channels']
#     new_model = DETECTORS.build(
#         cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
#
#     if torch.cuda.is_available():
#         new_model.cuda()
#     new_model = fuse_compactor(new_model, final_dict)
#     model_dict = new_model.state_dict()
#     checkpoint_model = checkpoint['state_dict']
#     checkpoint_pre_model_final = {k: v for k, v in checkpoint_model.items() if
#                                   'bbox_head' in k}
#     # or 'gt_backbone' in k or 'pointnet' in k or 'gt_attention' in k
#     checkpoint_model = {k: v for k, v in checkpoint_pre_model_final.items() if k in model_dict}
#     model_dict.update(checkpoint_model)
#     new_model.load_state_dict(model_dict)
#
#     return new_model
#
#
# def fuse_compactor(module, save_dict):
#     # new_model_kernel_name = []
#     # new_model_save_dict = {}
#     #
#     # for k, v in module.state_dict().items():
#     #     v = v.detach().cpu().numpy()
#     #     if v.ndim in [2, 4]:
#     #         new_model_kernel_name.append(k)
#     #     new_model_save_dict[k] = v
#     for name, layer in module.named_modules():
#         if isinstance(layer, nn.Conv2d) and 'backbone' in name:
#             layer.weight = nn.Parameter(torch.tensor(save_dict[f'{name}' + '.weight']).type(torch.FloatTensor))
#             layer.bias = nn.Parameter(torch.tensor(save_dict[f'{name}' + '.bias']).type(torch.FloatTensor))
#         if hasattr(layer, 'bn') and 'backbone' in name:
#             layer.bn = nn.Identity()
#     # fuse_identify_bn(module)
#
#     #     if 'conv' in name:
#     # for child_module in module.modules():
#     #     if hasattr(child_module, 'conv') and 'backbone' in child_module:
#     #         child_module.conv.weight = save_dict[f'{child_module}'+'.conv.weight']
#     #         child_module.conv.bias = save_dict[f'{child_module}'+'.conv.bias']
#
#     # last_conv = None
#     # last_conv_name = None
#     # for name, child in module.named_children():
#     #     if isinstance(child,
#     #                   (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
#     #         if last_conv is None:  # only fuse BN that is after Conv
#     #             continue
#     #         last_conv.weight = save_dict[last_conv_name.weight]
#     #         last_conv.bias = save_dict[last_conv_name.bias]
#     #         module._modules[last_conv_name] = last_conv
#     #         # To reduce changes, set BN as Identity instead of deleting it.
#     #         module._modules[name] = nn.Identity()
#     #         last_conv = None
#     #     elif isinstance(child, nn.Conv2d):
#     #         last_conv = child
#     #         last_conv_name = name
#     #     else:
#     #         fuse_compactor(child, save_dict)
#     return module
#
#
# def fuse_identify_bn(module):
#     """Recursively fuse conv and bn in a module.
#
#     During inference, the functionary of batch norm layers is turned off
#     but only the mean and var alone channels are used, which exposes the
#     chance to fuse it with the preceding conv layers to save computations and
#     simplify network structures.
#
#     Args:
#         module (nn.Module): Module to be fused.
#
#     Returns:
#         nn.Module: Fused module.
#     """
#     last_conv = None
#     last_conv_name = None
#
#     for name, child in module.named_children():
#         if isinstance(child,
#                       (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
#             if last_conv is None:  # only fuse  backbone BN that is after Conv
#                 continue
#             # To reduce changes, set BN as Identity instead of deleting it.
#             module._modules[name] = nn.Identity()
#             last_conv = None
#         elif isinstance(child, nn.Conv2d):
#             last_conv = child
#         else:
#             fuse_identify_bn(child)
#     return module
#
#
# def fuse_conv_bn(save_dict, pop_name_set, kernel_name):
#     mean_name = kernel_name.replace('.conv.weight', '.bn.running_mean')
#     var_name = kernel_name.replace('.conv.weight', '.bn.running_var')
#     gamma_name = kernel_name.replace('.conv.weight', '.bn.weight')
#     beta_name = kernel_name.replace('.conv.weight', '.bn.bias')
#     pop_name_set.add(mean_name)
#     pop_name_set.add(var_name)
#     pop_name_set.add(gamma_name)
#     pop_name_set.add(beta_name)
#     mean = save_dict[mean_name]
#     var = save_dict[var_name]
#     gamma = save_dict[gamma_name]
#     beta = save_dict[beta_name]
#     kernel_value = save_dict[kernel_name]
#     # print('kernel name', kernel_name)
#     # print('kernel, mean, var, gamma, beta', kernel_value.shape, mean.shape, var.shape, gamma.shape, beta.shape)
#     return _fuse_kernel(kernel_value, gamma, var, eps=1e-5), _fuse_bias(mean, var, gamma, beta, eps=1e-5)
#
#
# def _fuse_kernel(kernel, gamma, running_var, eps):
#     print('fusing: kernel shape', kernel.shape)
#     std = np.sqrt(running_var + eps)
#     t = gamma / std
#     t = np.reshape(t, (-1, 1, 1, 1))
#     print('fusing: t', t.shape)
#     t = np.tile(t, (1, kernel.shape[1], kernel.shape[2], kernel.shape[3]))
#     return kernel * t
#
#
# def _fuse_bias(running_mean, running_var, gamma, beta, eps, bias=None):
#     if bias is None:
#         return beta - running_mean * gamma / np.sqrt(running_var + eps)
#     else:
#         return beta + (bias - running_mean) * gamma / np.sqrt(running_var + eps)
#
#
# def fold_conv(fused_k, fused_b, thresh, compactor_mat):
#     metric_vec = np.sqrt(np.sum(compactor_mat ** 2, axis=(1, 2, 3)))
#     filter_ids_below_thresh = np.where(metric_vec < thresh)[0]
#
#     if len(filter_ids_below_thresh) == len(metric_vec):
#         sortd_ids = np.argsort(metric_vec)
#         filter_ids_below_thresh = sortd_ids[:-1]  # TODO preserve at least one filter
#
#     if len(filter_ids_below_thresh) > 0:
#         compactor_mat = np.delete(compactor_mat, filter_ids_below_thresh, axis=0)
#
#     kernel = F.conv2d(torch.from_numpy(fused_k).permute(1, 0, 2, 3), torch.from_numpy(compactor_mat),
#                       padding=(0, 0)).permute(1, 0, 2, 3)
#     Dprime = compactor_mat.shape[0]
#     bias = np.zeros(Dprime)
#     for i in range(Dprime):
#         bias[i] = fused_b.dot(compactor_mat[i, :, 0, 0])
#
#     if type(bias) is not np.ndarray:
#         bias = np.array([bias])
#     kernel = nn.Parameter(kernel)
#     bias = nn.Parameter(torch.from_numpy(bias))
#
#     return kernel, bias, filter_ids_below_thresh
#
#
# # def add_params(params, module, prefix='', is_dcn_module=None):
# #     is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))
# #     is_dwconv = (isinstance(module, torch.nn.Conv2d)
# #                  and module.in_channels == module.groups)
# #     for name, param in module.named_parameters(recurse=False):
# #         if
# #         params_group = {'params': [param]}
# #         if not param.requires_grad:
# #             params.append(params_group)
# #             continue
# #
# #
# # def compactor_mm(model):
# #     if hasattr(model, 'module'):
# #         model = model.module
# #
# #     compactor_paras = []
#
# def _fuse_conv_bn(conv, bn):
#     """Fuse conv and bn into one module.
#
#     Args:
#         conv (nn.Module): Conv to be fused.
#         bn (nn.Module): BN to be fused.
#
#     Returns:
#         nn.Module: Fused module.
#     """
#     conv_w = conv.weight
#     conv_b = conv.bias if conv.bias is not None else torch.zeros_like(
#         bn.running_mean)
#
#     factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
#     conv.weight = nn.Parameter(conv_w *
#                                factor.reshape([conv.out_channels, 1, 1, 1]))
#     conv.bias = nn.Parameter((conv_b - bn.running_mean) * factor + bn.bias)
#     return conv
#
#
# # def convert_compactor2(module):
# #     last_conv = None
# #     last_conv_name = None
# #     last2_conv = None
# #     last2_conv_name = None
# #
# #     for name, child in module.named_children():
#
# def compactor_weight_zero(model):
#     for submodule in model.modules():
#         if hasattr(submodule, 'mask'):
#             metric_vec = torch.Tensor(submodule.get_metric_vector())
#             filter_ids_below_thresh = torch.where(metric_vec < 1e-5)[0]
#             submodule.set_weight_zero(filter_ids_below_thresh)
#     return model
#
#
# def compactor_convert2(module, checkpoints):
#     new_channels, pruned_ids = get_deps_if_prune_low_metric(module)
#
#     # according to pruned_ids to create new model
#     cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
#     cfg_model = cfg.model
#     cfg_model.backbone.sa_channels = new_channels['sa_channels']
#     cfg_model.backbone.fp_channels = new_channels['fp_channels']
#     new_model = DETECTORS.build(
#         cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
#     if torch.cuda.is_available():
#         new_model.cuda()
#
#     compactor_layer = []
#     for child_module in module.modules():
#         if hasattr(child_module, 'compactor'):
#             compactor_layer.append(child_module)
#     i = 0
#     for m in range(len(compactor_layer)):
#         compactor_layer[i].conv.weight, compactor_layer[i].conv.bias, _ = fold_conv(
#             compactor_layer[i].conv.weight.detach().cpu().numpy(),
#             compactor_layer[i].conv.bias.detach().cpu().numpy(),
#             THRESH,
#             compactor_layer[i].compactor.pwc.weight.detach().cpu().numpy())
#         i += 1
#     kernel_name_list = []
#     fully_kernel_name_list = []
#     save_dict = {}
#     for k, v in module.state_dict().items():
#         if 'cof_attention' not in k:
#             v = v.detach().cpu().numpy()
#             if v.ndim in [2, 4] and 'compactor.pwc' not in k:
#                 kernel_name_list.append(k)
#                 fully_kernel_name_list.append(k)
#             elif v.ndim in [2, 4]:
#                 fully_kernel_name_list.append(k)
#             save_dict[k] = v
#     compactor_next_layer = []
#
#     for i in range(len(fully_kernel_name_list)):
#         if 'compactor' in fully_kernel_name_list[i]:
#             compactor_next_layer.append(fully_kernel_name_list[i + 1])
#         i += 1
#
#     for i in range(len(compactor_next_layer)):
#         save_dict[compactor_next_layer[i]] = np.delete(save_dict[compactor_next_layer[i]], pruned_ids[i], axis=1)
#
#     # m
#     for i in range(len(pruned_ids)):
#         if i == 3:
#             save_dict[fully_kernel_name_list[23]] = np.delete(save_dict[fully_kernel_name_list[23]], pruned_ids[i],
#                                                               axis=1)
#         if i == 5:
#             save_dict[fully_kernel_name_list[20]] = np.delete(save_dict[fully_kernel_name_list[20]], pruned_ids[i],
#                                                               axis=1)
#         if i == 7:
#             save_dict[fully_kernel_name_list[20]] = np.delete(save_dict[fully_kernel_name_list[20]],
#                                                               pruned_ids[i] + 256, axis=1)
#         if i == 8:
#             save_dict[fully_kernel_name_list[23]] = np.delete(save_dict[fully_kernel_name_list[23]],
#                                                               pruned_ids[i] + 256, axis=1)
#     new_state_dict = {}
#     for n_k, n_v in new_model.state_dict().items():
#         if 'bn' not in n_k:
#             new_state_dict[n_k] = torch.from_numpy(save_dict[n_k])
#     new_model_state = new_model.state_dict()
#     new_model_state.update(new_state_dict)
#     new_model.load_state_dict(new_model_state)
#     new_model = fuse_identify_bn(new_model)
#     return new_model
