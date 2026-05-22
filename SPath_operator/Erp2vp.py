import torch
import torch.nn as nn
import SPath
import numpy as np
from SPath_operator.BaseOpModule import BaseOpModule

class Erp2vp_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, rota, op):
        gid = x.device.index
        x = x.contiguous()
        rota = rota.contiguous()
        outputs = op[gid].forward(x, rota)
        ctx.op = op
        ctx.save_for_backward(rota,outputs[1],outputs[2])
        return outputs[0]
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        gid = grad_output.device.index
        rota,_,_ = ctx.saved_tensors
        outputs = ctx.op[gid].backward(grad_output, rota)
        return outputs[0], None, None
    

class Erp2vp(BaseOpModule):
    
    def __init__(self, fov,h_v,w_v, device = 0, time_it = False):
        super(Erp2vp, self).__init__(device)
        self.op = { gid : SPath.Erp2vpOp(fov,h_v,w_v, gid, time_it) for gid in self.device_list}
        self.pi = np.pi
        

    def forward(self, x, rota):
        x_deg = x / 180 * self.pi
        res = Erp2vp_AF.apply(x_deg, rota, self.op)
        return res
