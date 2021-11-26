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

    def __init__(self, quantize=True, bit=2):
        self.bit = bit
        self.quantize = quantize

    def pack_hook(self, x):
        if self.quantize:
            return self.controller.quantize(x)
        return x

    def unpack_hook(self, x):
        if self.quantize:
            return self.controller.dequantize(x)
        return x

    def before_run(self, runner):
        self.controller = actnn.controller.Controller(
            default_bit=self.bit)
        model = (runner.model.module if is_module_wrapper(
            runner.model) else runner.model)
        self.controller.filter_tensors(model.named_parameters())
        torch._C._autograd._register_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook)

    def after_run(self, runner):
        torch._C._autograd._reset_saved_tensors_default_hooks()

    def after_train_iter(self, runner):
        if self.quantize:
            self.controller.iterate()
