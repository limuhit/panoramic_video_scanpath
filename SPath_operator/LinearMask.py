import torch
import torch.nn as nn
import SPath
from SPath_operator.BaseOpModule import BaseOpModule



class LinearMask(BaseOpModule):

    def __init__(self, ncontext, npred, cin, cout, hidden=True, out=False, device = 0, time_it = False):
        super(LinearMask, self).__init__(device)
        ngroup = ncontext + npred
        self.op = { gid : SPath.LinearMaskOp(ncontext,npred,cin,cout,hidden, out, gid, time_it) for gid in self.device_list}
        if out:
            self.weight = nn.Parameter(torch.empty((cout*npred,cin*ngroup),dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(cout*npred,dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.empty((cout*ngroup,cin*ngroup),dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(cout*ngroup,dtype=torch.float32))
        torch.nn.init.kaiming_normal_(self.weight)
        self.mask, self.init = None, False
        self.mod, self.nctx, self.pid = npred, ncontext, 0
        self.cout, self.fout = cout, out
        self.on,self.ow = -1, -1

    def setup_mask(self,x):
        gid = x.device.index
        self.mask = self.op[gid].forward()
        self.init = True

    def forward(self, x):
        if not self.init: self.setup_mask(x)
        weight = self.weight*self.mask
        res = nn.functional.linear(x,weight,self.bias)
        return res

    def prepare(self,x):
        if not self.init: self.setup_mask(x)
        n,w = x.shape
        if not(self.on == n and self.ow == w):
            pbase = 0 if self.fout else self.nctx
            self.out = torch.zeros((n,(pbase+self.mod)*self.cout),dtype=torch.float32,device=x.device)
        if not (self.out.device.index == x.device.index):
            self.out = self.out.to(x.device)
        return self.out

    def forward_pred(self,x):
        pbase = 0 if self.fout else self.nctx
        if self.pid == 0:
            ostart, oend = 0, (pbase+1)*self.cout
        else:
            ostart, oend = (pbase+self.pid)*self.cout, (pbase+self.pid+1)*self.cout
        self.pid = (self.pid + 1) % self.mod
        tw = self.weight[ostart:oend]*self.mask[ostart:oend]
        tb = self.bias[ostart:oend]
        res = nn.functional.linear(x,tw,tb)
        self.out[:,ostart:oend] = res
        return self.out
