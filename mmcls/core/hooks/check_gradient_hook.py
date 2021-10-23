# # Copyright (c) OpenMMLab. All rights reserved.
# # https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/checkloss_hook.py
# import numpy as np
# from mmcv.parallel import is_module_wrapper
# from mmcv.runner.hooks import HOOKS, Hook
# import torch
# import pickle

# @HOOKS.register_module()
# class CheckGradientHook(Hook):
#     """.
#     This hook will record the weight gradient
#     Args:
#         actnn: if quantize, actnn=True, otherwise False
#         interval (int): Record interval (every k iterations)
#         minibatch (bool): If current run is a minibatch
#     """

#     def __init__(self, minibatch, interval=1):
#         self.interval = interval

#         self.num_batches = 10
#         self.batch_gradient = None
#         self.actnn_gradient = None
#         self.cal_batch_gradient = True
#         self.cal_actnn_gradient = False

#     def after_train_iter(self, runner):
#         gradient = None
#         model = runner.model.module if is_module_wrapper(runner.model) else runner.model
#         for param in model.parameters():
#             if param.requires_grad and param.grad is not None:
#                 cur_param = param.grad.reshape(1, -1)
#                 if gradient is None:
#                     gradient = cur_param
#                 else:
#                     gradient = torch.cat((gradient, cur_param), 1)

#         if self.cal_batch_gradient:
#             if self.batch_gradient is None:
#                 self.batch_gradient = gradient
#             else:
#                 self.batch_gradient += gradient

#         if self.cal_actnn_gradient:
#             if self.actnn_gradient is None:
#                 self.actnn_gradient = gradient
#             else:
#                 self.actnn_gradient += gradient

#         if self.every_n_iters(runner, self.num_batches):
#             pass


# Copyright (c) OpenMMLab. All rights reserved.
# https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/hook/checkloss_hook.py
import numpy as np
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
import torch
import pickle


@HOOKS.register_module()
class CheckGradientHook(Hook):
    """.
    This hook will record the weight gradient
    Args:
        actnn: if quantize, actnn=True, otherwise False
        interval (int): Record interval (every k iterations)
        minibatch (bool): If current run is a minibatch
    """

    def __init__(self, minibatch, interval=1):
        self.interval = interval
        self.minibatch = minibatch
        self.gradient = None
        self.gradients = []

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, 1):
            gradient = None
            model = (
                runner.model.module if is_module_wrapper(runner.model) else runner.model
            )
            for param in model.parameters():
                if param.grad is not None:
                    cur_param = param.grad.reshape(1, -1)
                    if gradient is None:
                        gradient = cur_param
                    else:
                        gradient = torch.cat((gradient, cur_param), 1)
            if self.gradient is None:
                self.gradient = gradient
            else:
                self.gradient += gradient
            self.gradients.append(gradient)

        if self.every_n_iters(runner, self.interval):
            mean_grad = self.gradient / self.interval
            var = None
            for i in range(self.interval):
                if var is None:
                    var = (self.gradients[i] - mean_grad) ** 2
                else:
                    var += (self.gradients[i] - mean_grad) ** 2
            print("Gradient Variance", var.sum() / self.interval)
            # exit(0)

            if runner.actnn:
                prefix = "actnn"
            else:
                prefix = "no_actnn"
            if self.minibatch:
                prefix += "_minibatch"
            else:
                assert self.interval == 1
                prefix += "_batch"
            filename = "gradients/%s_gradient.p" % (prefix)
            print("Save gradient to ", filename)
            with open(filename, "wb") as f:
                pickle.dump(self.gradient, f)
            exit(0)
