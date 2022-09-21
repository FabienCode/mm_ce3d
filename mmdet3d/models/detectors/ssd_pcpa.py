# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models import DETECTORS
from .votenet_pcpa import Votepcpa


@DETECTORS.register_module()
class SSD3DNet_pcpa(Votepcpa):
    """3DSSDNet model.

    https://arxiv.org/abs/2002.10187.pdf
    """

    def __init__(self,
                 backbone,
                 label_backbone=None,
                 gt_backbone=None,
                 gt_attention=None,
                 pointnet=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(SSD3DNet_pcpa, self).__init__(
            backbone=backbone,
            label_backbone=label_backbone,
            gt_backbone=gt_backbone,
            gt_attention=gt_attention,
            pointnet=pointnet,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            pretrained=pretrained)
