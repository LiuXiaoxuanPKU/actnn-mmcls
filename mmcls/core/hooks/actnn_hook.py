# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/checkloss_hook.py
import numpy as np
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
import torch
import actnn

@HOOKS.register_module()
class ActnnHook(Hook):
    """.
    This hook will call actnn controller.iterate, which is used to update auto precision
    Args:
        actnn (bool): If actnn is enabled
        interval (int): Update interval (every k iterations)
    """

    def __init__(self, bit=2, auto_prec=False, interval=1):
        self.interval = interval
        self.bit = bit
        self.auto_prec = auto_prec

    def pack_hook(self, x):
        return self.controller.quantize(x)

    def unpack_hook(self, x):
        return self.controller.dequantize(x)

    def before_run(self, runner):
        self.controller = actnn.controller.Controller(
            default_bit=self.bit, auto_prec=self.auto_prec)
        model = (runner.model.module if is_module_wrapper(
                    runner.model) else runner.model)
        self.controller.filter_tensors(model.named_parameters())
        torch._C._autograd._register_saved_tensors_default_hooks(self.pack_hook, self.unpack_hook) 

    def after_run(self, runner):
        torch._C._autograd._reset_saved_tensors_default_hooks()

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            model = (
                runner.model.module if is_module_wrapper(
                    runner.model) else runner.model
            )
            self.controller.iterate(model)
