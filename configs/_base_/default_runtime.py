checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
# load_from = "/data2/yaoming/fcaf3d/log/votenet/scannet/pcpa_2.27_v1_step1/epoch_33.pth"
# load_from = "/data/private/hym/project/fcaf3d_midea/log/checkpoints/votenet_sunrgbd_gtpc_labelpcbackbone_pointNetw2_backbone.pth"

# load_from = "/data/private/hym/project/fcaf3d_midea/tools/pruned_fp1_0.3_votenet.pth"
# load_from = "/data/private/hym/project/fcaf3d_midea/log/checkpoints/gp3d_scannet_w2_labelbackbone.pth"
# load_from = "/data/private/hym/project/fcaf3d_midea/log/votenet_pruning/scannet/ori_votenet_4.17_v1/epoch_36.pth"
# load_from = "/data/private/hym/project/fcaf3d_midea/pruning_log/votenet/scannet/ori_3.19_v1/latest.pth"
# load_from = "/data/private/hym/project/fcaf3d_midea/log/epoch_36.pth"
load_from = None
resume_from = None
workflow = [('train', 1)]
