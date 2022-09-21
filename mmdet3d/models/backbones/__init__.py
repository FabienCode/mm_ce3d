# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG, RR_PointNet2SASSG, CoF_PointNet2SASSG
from .second import SECOND
# from .resrep_pointnet2_ssg import RR_PointNet2SASSG
from .cof_pointnet2_ssg import cof_PointNet2SASSG
from .cof_rr_pointnet2 import cof_rr_PointNet2SASSG
from .dis_resrep_pointnet import dis_rr_PointNet2SASSG
from .cd_rr_pointnet2_ssg import cd_rr_PointNet2SASSG
from .attention import teacher_attention, pointnet

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'PointNet2SASSG', 'PointNet2SAMSG', 'MultiBackbone', 'RR_PointNet2SASSG', 'CoF_PointNet2SASSG',
    'cof_PointNet2SASSG', 'cof_rr_PointNet2SASSG', 'dis_rr_PointNet2SASSG', 'cd_rr_PointNet2SASSG', 'teacher_attention',
    'pointnet'
]
