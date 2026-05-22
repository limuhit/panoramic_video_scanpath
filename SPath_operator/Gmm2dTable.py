import torch
import torch.nn as nn
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class Gmm2dTable_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, wt, mean, std, op):
        gid = wt.device.index
        wt=wt.contiguous()
        mean=mean.contiguous()
        std=std.contiguous()
        outputs = op[gid].forward(wt,mean,std)
        ctx.op = op
        return outputs[0]
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None
    

class Gmm2dTable(BaseOpModule):
    
    def __init__(self, num_gaussian, hboard, wboard, stride_h, stride_w,device = 0, time_it = False):
        super(Gmm2dTable, self).__init__(device)
        self.op = { gid : SPath.Gmm2dTableOp(num_gaussian, hboard, wboard, stride_h, stride_w, gid, time_it) for gid in self.device_list}
        

    def forward(self, wt, mean, std):
        res = Gmm2dTable_AF.apply(wt, mean, std, self.op)
        return res
