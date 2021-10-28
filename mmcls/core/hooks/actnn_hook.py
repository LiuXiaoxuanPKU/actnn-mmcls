# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/checkloss_hook.py
import numpy as np
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
import torch
import pickle


@HOOKS.register_module()
class ActnnHook(Hook):
    """.
    This hook will call actnn controller.iterate, which is used to update auto precision
    Args:
        actnn (bool): If actnn is enabled
        interval (int): Update interval (every k iterations)
    """

    def __init__(self, interval=1):
        self.interval = interval

    def after_train_iter(self, runner):
        if runner.actnn:
            if self.every_n_iters(runner, self.interval):
                model = (
                    runner.model.module if is_module_wrapper(
                        runner.model) else runner.model
                )
                runner.controller.iterate(model)
