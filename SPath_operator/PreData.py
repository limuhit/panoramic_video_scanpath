import torch
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class PreData_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, pt, randv, start_idx, op):
        gid = x.device.index
        outputs = op[gid].forward(x,pt,randv,start_idx)
        ctx.op = op
        return outputs[0],outputs[1],outputs[2]

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None


class PreData(BaseOpModule):
    def __init__(self, nw, windows, npred, stride, train, device = 0, pathw=61, x_bias=223.5, y_bias=125.5, time_it = False):
        super(PreData, self).__init__(device)
        self.stride, self.nw, self.windows, self.npred = stride, nw, windows, npred
        self.train_f = train
        self.op = { gid : SPath.PreDataOp(x_bias, y_bias, pathw, nw, windows, npred, stride, train, gid, time_it) for gid in self.device_list}

    def forward(self, x, pt):
        num = (x.shape[1] - self.windows - self.npred - 1) // self.stride + 1
        if self.train_f:
            randv = torch.randperm(num-1,generator=None,device=x.device,dtype=torch.int32).contiguous()
            start_idx = torch.randint(0,self.stride,(x.shape[0],),device=x.device,dtype=torch.int32).contiguous()
        else:
            randv = torch.arange(0,num,device=x.device,dtype=torch.int32).contiguous()
            start_idx = torch.zeros((x.shape[0]),device=x.device,dtype=torch.int32).contiguous()
        res = PreData_AF.apply(x, pt, randv, start_idx, self.op)
        return res[0],res[1],res[2]


class PreData2_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, pt, pt_raw, randv, start_idx, op):
        gid = x.device.index
        outputs = op[gid].forward2(x,pt,pt_raw,randv,start_idx)
        ctx.op = op
        return outputs[0],outputs[3],outputs[2]

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None, None


class PreData2(BaseOpModule):
    def __init__(self, nw, windows, npred, stride, train, device = 0, pathw=61, x_bias=223.5, y_bias=125.5, time_it = False):
        super(PreData2, self).__init__(device)
        self.stride, self.nw, self.windows, self.npred = stride, nw, windows, npred
        self.train_f = train
        self.op = { gid : SPath.PreDataOp(x_bias, y_bias, pathw, nw, windows, npred, stride, train, gid, time_it) for gid in self.device_list}

    def forward(self, x, pt, pt_raw):
        num = (x.shape[1] - self.windows - self.npred - 1) // self.stride + 1
        if self.train_f:
            randv = torch.randperm(num-1,generator=None,device=x.device,dtype=torch.int32).contiguous()
            start_idx = torch.randint(0,self.stride,(x.shape[0],),device=x.device,dtype=torch.int32).contiguous()
        else:
            randv = torch.arange(0,num,device=x.device,dtype=torch.int32).contiguous()
            start_idx = torch.zeros((x.shape[0]),device=x.device,dtype=torch.int32).contiguous()
        res = PreData2_AF.apply(x, pt, pt_raw, randv, start_idx, self.op)
        return res[0],res[1],res[2]
