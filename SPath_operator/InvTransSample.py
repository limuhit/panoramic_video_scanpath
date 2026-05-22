import torch
import torch.nn as nn
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class InvTransSample_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, rand_v, old_p, op):
        x = x.contiguous()
        rand_v = rand_v.contiguous()
        old_p = old_p.contiguous()
        gid = x.device.index
        outputs = op[gid].forward(x, rand_v,old_p)
        ctx.op = op
        return outputs[1],outputs[2]

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None


class InvTransSampleThreshold(BaseOpModule):

    def __init__(self, nsample, stride_h, stride_w, threshold= 1e-5, device = 0, time_it = False):
        super(InvTransSampleThreshold, self).__init__(device)
        self.nsample = nsample
        self.op = { gid : SPath.InvtranssampleOp(nsample, stride_h, stride_w, gid, time_it) for gid in self.device_list}
        self.threshold = threshold


    def forward(self, x, old_p):
        num = x.shape[0]
        rand_v = torch.rand((num,self.nsample),device=x.device,dtype=torch.float32)
        loc,pval = InvTransSample_AF.apply(x, rand_v, old_p, self.op)
        ps = self.op[x.device.index].select_thr(old_p,self.threshold)
        return ps[0], ps[1], ps[2]

class InvTransSample(BaseOpModule):

    def __init__(self, inner_group, nsample, stride_h, stride_w, device = 0, time_it = False):
        super(InvTransSample, self).__init__(device)
        self.nsample = nsample
        self.one_pass = nsample < 2
        self.inner_group = inner_group
        self.op = { gid : SPath.InvtranssampleOp(nsample, stride_h, stride_w, gid, time_it) for gid in self.device_list}


    def forward(self, x, old_p):
        num = x.shape[0]
        rand_v = torch.rand((num,self.nsample),device=x.device,dtype=torch.float32)
        loc,pval = InvTransSample_AF.apply(x, rand_v, old_p, self.op)
        if self.one_pass:
            return None, loc.view(num,2), pval.view(num)
        if self.inner_group:
            value, idx = torch.topk(pval,k=1,dim=1,largest=False)
            pbase = torch.arange(0,num,device=x.device,dtype=torch.int32)*self.nsample
            value, idx = value.view(num).contiguous(),idx.view(num).type(torch.int32).contiguous()
            idx += pbase
        else:
            value,idx = torch.topk(pval.view(-1),k=num,largest=False)
            value, idx = value.view(num).contiguous(),idx.view(num).type(torch.int32).contiguous()
        ps = self.op[x.device.index].select(idx)
        return ps[0], ps[1], ps[2]

class Softmax2DM(nn.Module):

    def __init__(self):
        super(Softmax2DM, self).__init__()
        self.op = nn.Softmax(dim=1)

    def forward(self, x):
        n,c,h,w = x.shape
        y = self.op(x.view(n,c*h*w))
        return y.view(n,c,h,w).contiguous()
