_base_ = [
    '../_base_/models/vgg11bn.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
actnn = True
data = dict(
    samples_per_gpu=64,  # 64*4 = 256
    workers_per_gpu=2,
)

check_gradient = True
# # do not update weights to check gradient
# optimizer_config = dict(grad_clip=None, update_interval=10000)

log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project='classification',
                entity='actnn',
                name='vgg11bn_b64x4_imagenet',
            )
        )
    ]
)


custom_hooks = [
    dict(
        type="ActnnHook",
        interval=1
    )
]
