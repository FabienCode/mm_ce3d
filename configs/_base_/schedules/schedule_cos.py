# optimizer
# This schedule is mainly used by models in indoor dataset,
# e.g. BRNet on SUNGRBD and ScanNet
lr = 0.008  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0
)
# runtime settings
runner = dict(type='RR_EpochBasedRunner', max_epochs=72)
