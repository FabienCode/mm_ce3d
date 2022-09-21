import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from mmcv.cnn import get_model_complexity_info
from mmdet3d.models.builder import DETECTORS
from mmcv import Config

def rr_flops(print_log=True):
    cfg = Config.fromfile('/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py')
    cfg_model = cfg.model
    # cfg_model.backbone.sa_channels = channel['sa_channels']
    # cfg_model.backbone.fp_channels = channel['fp_channels']
    tmp_model = DETECTORS.build(
        cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
    if torch.cuda.is_available():
        tmp_model.cuda()
    tmp_model.eval()


    if hasattr(tmp_model, 'forward_dummy'):
        tmp_model.forward = tmp_model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported for {}'.format(
                tmp_model.__class__.__name__))
    # temp_model2 = copy.deepcopy(tmp_model)
    input_shape = (40000, 3)
    flops, params = get_model_complexity_info(tmp_model, input_shape, print_per_layer_stat=print_log)
    # flops1 = round((float(flops1.split()[0]) - 9.248), 2)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops is: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    return flops, params

if __name__ == '__main__':
    rr_flops()
