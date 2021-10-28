_base_ = [
    '../_base_/models/vgg11bn.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]
actnn = True
bit = 8
auto_prec = True

data = dict(
    samples_per_gpu=8,  # 64*4 = 256
    workers_per_gpu=2,
)

# checkpoint every 5k iter
checkpoint_config = dict(interval=5000, by_epoch=False)
load_from = "/home/ubuntu/actnn-mmcls/work_dirs/vgg11bn_b64x4_imagenet_new/epoch_1.pth"
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
    dict(type="CheckGradientHook", interval=100),
    dict(
        type="ActnnHook",
        interval=1
    )
]
