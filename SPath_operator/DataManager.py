import torch
import SPath
from SPath_operator.BaseOpModule import BaseOpModule

class DataManager(BaseOpModule):

    def __init__(self, nsample, nlayer, device = 0, time_it = False):
        super(DataManager, self).__init__(device)
        self.op = { gid : SPath.DataManagerOp(nlayer, gid, time_it) for gid in self.device_list}
        self.old_shape = None
        self.old_device = None
        self.temp = None
        self.one_pass = nsample < 2

    def check(self, x:torch.Tensor):
        modified = False
        if not (self.old_shape is None):
            if not (self.old_shape == x.shape and self.old_device == x.device):
                modified = True
        else:
            modified = True
        self.old_shape, self.old_device = x.shape, x.device
        if modified:
            gid = x.device.index
            self.op[gid].clear()
        return modified

    def push(self,x:torch.Tensor):
        gid = x.device.index
        self.op[gid].push(x)

    def setup_template(self,x):
        modified = self.temp is None
        if not modified:
            if not((x.shape[0] ==self.temp.shape[0]) and (x.device == self.temp.device)):
                modified = True
        if modified:
            self.temp = torch.arange(0,x.shape[0],device=x.device,dtype=torch.float32)

    def forward(self, x):
        if self.one_pass: return x
        self.setup_template(x)
        diff = torch.sum(torch.abs(self.temp-x)).item()
        if diff > 1e-5:
            gid = x.device.index
            res = self.op[gid].forward(x)
            return res
        else:
            return x
