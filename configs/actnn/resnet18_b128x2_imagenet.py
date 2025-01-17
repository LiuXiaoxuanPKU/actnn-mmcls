_base_ = [
    '../_base_/models/resnet18.py',
    '../_base_/datasets/imagenet_bs32.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
actnn = True
data = dict(
    samples_per_gpu=128, # 128*2 = 256
    workers_per_gpu=1,
)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='classification',
                entity='actnn',
                name='resnet18_b128x2_imagenet',
            )
        )
    ]
)
