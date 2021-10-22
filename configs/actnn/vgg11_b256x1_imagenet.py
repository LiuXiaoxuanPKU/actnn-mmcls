_base_ = [
    '../_base_/models/vgg11.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
actnn = True
optimizer = dict(lr=0.01)
data = dict(
    samples_per_gpu=256, # 256 * 1 = 256
    workers_per_gpu=8,
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
                name='vgg11_b256x1_imagenet',
            )
        )
    ]
)
