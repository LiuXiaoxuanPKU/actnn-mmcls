_base_ = [
    '../_base_/models/vgg11bn.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py',
    '../_base_/default_runtime.py'
]

data = dict(
    # samples_per_gpu=4,  # 64*4 = 256
    samples_per_gpu=64,
    workers_per_gpu=2,
)

# checkpoint every 5k iter
checkpoint_config = dict(interval=5000, by_epoch=False)
# load from checkpoint to test the correctness of auto precision
load_from = "/home/ubuntu/actnn-mmcls/work_dirs/vgg11bn_b64x4_imagenet_new/epoch_1.pth"

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

actnn = True
bit = 4
custom_hooks = [
    # dict(type="RecordGradientHook", interval=1000),
    dict(
        type="ActnnHook",
        quantize=actnn,
        bit=bit,
    )
]
