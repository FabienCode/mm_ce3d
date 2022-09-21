from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from .base import Base3DDetector


@DETECTORS.register_module()
class LabelPrior(Base3DDetector):
    """SingleStage3DDetector.

    This class serves as a base class for single-stage 3D detectors.

    Args:
        backbone (dict): Config dict of detector's backbone.
        neck (dict, optional): Config dict of neck. Defaults to None.
        bbox_head (dict, optional): Config dict of box head. Defaults to None.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        pretrained (str, optional): Path of pretrained models.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 label_backbone=None,
                 gt_backbone=None,
                 gt_attention=None,
                 pointnet=None,
                 neck=None,
                 label_neck=None,
                 gt_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None):
        super(LabelPrior, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        if label_backbone is not None:
            self.label_backbone = build_backbone(label_backbone)
        if gt_backbone is not None:
            self.gt_backbone = build_backbone(gt_backbone)
        if gt_attention is not None:
            self.gt_attention = build_backbone(gt_attention)
        if pointnet is not None:
            self.pointnet = build_backbone(pointnet)
        if neck is not None:
            self.neck = build_neck(neck)
        if label_neck is not None:
            self.label_neck = build_neck(label_neck)
        if gt_neck is not None:
            self.gt_neck = build_neck(gt_neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, points, img_metas=None):
        """Directly extract features from the backbone+neck.

        Args:
            points (torch.Tensor): Input points.
        """
        x = self.backbone(points)
        if self.with_neck:
            x = self.neck(x)
        return x

    def extract_feats(self, points, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, img_meta)
            for pts, img_meta in zip(points, img_metas)
        ]
