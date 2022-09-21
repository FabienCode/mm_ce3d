# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 0.004  # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
                 # paramwise_cfg=dict(
                 #     custom_keys={
                 #         'compactor': dict(decay_mult=0)
                 #     }))
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[48, 64])
# lr_config = dict(
#     policy="CosineAnnealing",
#     min_lr=0
# )
# runtime settings
runner = dict(type='RR_EpochBasedRunner', max_epochs=72)
