import torch

import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor
from mmdet.models.builder import DETECTORS


def main():
    cfg = Config.fromfile("/data/private/hym/project/fcaf3d_midea/configs/votenet/votenet_8x8_scannet-3d-18class.py")
    cfg_model = cfg.model
    # build the model and load checkpoint
    model = DETECTORS.build(
        cfg_model, default_args=dict(train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg')))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    dummy_input = torch.randn(4, 40000, 4).cuda()
    input_names = ["point clouds"]
    output_names = ["bbox"]
    torch.onnx.export(model, dummy_input, '/data/private/hym/project/fcaf3d_midea/log/test+pointnet.onnx')


if __name__ == '__main__':
    main()
