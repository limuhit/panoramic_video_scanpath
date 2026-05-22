import torch
import SPath
import numpy as np
from SPath_operator.BaseOpModule import BaseOpModule

class Viewport_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, theta_phi, op):
        x = x.contiguous()
        theta_phi = theta_phi.contiguous()
        gid = x.device.index
        outputs = op[gid].forward(x, theta_phi)
        ctx.op = op
        return outputs[0],outputs[4],outputs[2]

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None

class Viewport(BaseOpModule):
    """Project ERP frames to viewports at the given spherical coordinates."""

    def __init__(self, fov,h,w, device = 0, time_it = False):
        super(Viewport, self).__init__(device)
        self.op = { gid : SPath.ViewportOp(fov,h,w, gid, time_it) for gid in self.device_list}
        self.register_buffer('viewport_tf_filed', None)
        self.register_buffer('rota', None)
        self.pi = np.pi

    def forward(self, x, theta_phi):
        theta_phi = theta_phi / 180 * self.pi
        res = Viewport_AF.apply(x, theta_phi, self.op)
        self.viewport_tf_filed = res[1]
        self.rota = res[2]
        return res[0]

    def get_rota_matrix(self,theta_phi):
        theta_phi = theta_phi / 180 * self.pi
        return self.op[theta_phi.device.index].cal_rota_matrix(theta_phi)

    def get_viewport_xy(self, theta_phi_next):
        theta_phi_next = theta_phi_next / 180 * self.pi
        gid = theta_phi_next.device.index
        res = self.op[gid].get_viewport_xy(theta_phi_next)
        return res[0]
