import torch
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class GmmSample_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, wt,mean,std,r1,r2,op):
        wt,mean,std = wt.contiguous(),mean.contiguous(),std.contiguous()
        gid = wt.device.index
        outputs = op[gid].forward(wt,mean,std,r1,r2)
        ctx.op = op
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None


class GmmSample(BaseOpModule):

    def __init__(self, nout, thr_x, thr_y, device = 0, time_it = False):
        super(GmmSample, self).__init__(device)
        self.nout = nout
        self.th_x, self.th_y = thr_x, thr_y
        self.op = { gid : SPath.GmmSampleOp(nout, thr_x, gid, time_it) for gid in self.device_list}


    def forward(self, wt, mean, std):
        r1 = torch.rand((wt.shape[0],self.nout),dtype=torch.float32,device=wt.device).contiguous()
        r2 = torch.normal(0,1,(wt.shape[0],self.nout,2),dtype=torch.float32,device=wt.device).contiguous()
        res = GmmSample_AF.apply(wt,mean,std,r1,r2, self.op)
        res = self.op[wt.device.index].select(res,self.th_x,self.th_y)
        return res[1]
