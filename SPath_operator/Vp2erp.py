import torch
import torch.nn as nn
import numpy as np
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class Vp2erp_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, rota, op):
        gid = x.device.index
        outputs = op[gid].forward(x, rota)
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None
    

class Vp2erp(BaseOpModule):
    
    def __init__(self, fov,h_v,w_v, device = 0, time_it = False):
        super(Vp2erp, self).__init__(device)
        self.op = { gid : SPath.Vp2erpOp(fov,h_v,w_v, gid, time_it) for gid in self.device_list}
        self.pi = np.pi
        

    def forward(self, x, rota):
        res = Vp2erp_AF.apply(x, rota, self.op)
        res = res / self.pi * 180
        return res
