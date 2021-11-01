import numpy as np
from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook
import torch
import pickle
import random


@HOOKS.register_module()
class RecordGradientHook(Hook):
    """.
    This hook will record the weight gradient
    Args:
        interval (int): Record after training for k iterations
    """

    def __init__(self, interval=1):
        self.iter_cnt = 0
        self.interval = interval

    def after_train_iter(self, runner):
        self.iter_cnt += 1
        # if self.iter_cnt == self.interval - 1:
        #     seed = 0
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed_all(seed)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False

        if self.iter_cnt < self.interval:
            return
        gradient = None
        model = (
            runner.model.module if is_module_wrapper(
                runner.model) else runner.model
        )
        # gradient for current iteration
        for param in model.parameters():
            if param.grad is not None:
                cur_param = param.grad.reshape(1, -1)
                if torch.isnan(param.grad).any():
                    print("Param grad conatins nan")
                    print(param)
                    exit(0)
                if gradient is None:
                    gradient = cur_param
                else:
                    gradient = torch.cat((gradient, cur_param), 1)

        if runner.actnn and runner.auto_prec:
            filename = "actnn_autoprec.p"
        elif runner.actnn and not runner.auto_prec:
            filename = "actnn_%s.p" % runner.bit
        elif not runner.actnn:
            filename = "org.p"

        filename = "gradient/" + filename
        with open(filename, "wb") as f:
            pickle.dump(gradient, f)
        exit(0)
