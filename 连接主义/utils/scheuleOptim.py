"""from pytorch-bert"""
import numpy as np
from torch import nn

class  ScheduleOptim:
    """关于学习率变化的优化器的包装类"""
    def __init__(self,optimizer,lr,d_model):
        self._optimizer=optimizer
        self.lr=lr
        self.current_steps=0
        self.init_lr=np.power(d_model,-0.5)
    def step_and_update_lr(self):
        self._updating_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer._zero_grad()

    def _get_lr_scale(self):
        """参考对应学习率变化公式
        每步都根据
        """

        pass

    def _updating_learning_rate(self):
        """每一步更新优化器学习率"""
        self.current_steps+=1
        lr=self.init_lr*self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr']=lr

model=nn.Conv2d(24,40,(3,4))

