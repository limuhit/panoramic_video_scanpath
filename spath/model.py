import torch
from torch import nn

from SPath_operator import DataManager, GMM2D, Gmm2dTable, InvTransSampleThreshold

from spath.layers import ContextNet, LinearResBlock, PathNet, resnet50
from spath.serialization import safe_torch_load


class PIDNavigator:
    def __init__(self, kp, ki, kd, dt=1.0):
        self.reset()
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.dt2 = dt * dt
        self.idt = 1.0 / dt
        self.thr_x = 70
        self.thr_y = 30

    def reset(self):
        self.speed = None
        self.speed_inc = None
        self.loc = None
        self.old_err = None

    def Ziegler_Nichols(self, ku, pu):
        self.kp = 0.6 * ku
        self.ki = 2 * ku / pu
        self.kd = ku * pu / 8
        self.old_Ku, self.old_Pu = ku, pu

    def start_naviagor(self, x):
        self.speed = -x * self.idt
        self.loc = torch.zeros_like(x)
        self.speed_inc = torch.zeros_like(x)
        self.old_err = torch.zeros_like(x)
        self.err_sum = torch.zeros_like(x)
        self.abs_err = torch.zeros_like(x)

    def get_loc(self):
        dist = self.speed * self.dt + 0.5 * self.speed_inc * self.dt2
        dist[:, 0] = torch.clip_(dist[:, 0], -self.thr_x, self.thr_x)
        dist[:, 1] = torch.clip_(dist[:, 1], -self.thr_y, self.thr_y)
        self.loc = self.loc + dist
        self.speed = self.speed + self.speed_inc * self.dt
        return self.loc

    def pid(self, target):
        loc = self.get_loc()
        err = target - loc
        self.err_sum = self.err_sum + err * self.dt
        self.abs_err = self.abs_err + torch.abs(err)
        de = (err - self.old_err) * self.idt
        self.speed_inc = self.kp * err + self.ki * self.err_sum + self.kd * de
        self.old_err = err
        return loc


class ScanpathPredictor(nn.Module):
    """Core SPath model.

    The same module supports teacher-forced training through ``forward`` and
    autoregressive sampling through ``forward_base`` / ``forward_pred``.
    """

    def __init__(self, windows=5, npred=5, nsample=1, stride=0.2, ng=3, gid=0):
        super().__init__()
        self.npred = npred
        self.ng = ng
        self.vd_net = resnet50()
        self.pt_net = PathNet(windows, npred)
        self.ctx_net = ContextNet(npred, 2, 32, gid=gid)
        self.conv1 = nn.Conv2d(2048, 16, 1)
        self.conv2 = nn.Conv2d(windows, npred, 1)
        self.vd_pred_net = nn.Sequential(
            nn.Linear(112 * 16, 256),
            LinearResBlock(256, 256),
            LinearResBlock(256, 256),
            LinearResBlock(256, 256),
            nn.Linear(256, 128),
        )
        self.weight_net = nn.Sequential(
            nn.Linear(288, 64),
            LinearResBlock(64, 64),
            LinearResBlock(64, 64),
            nn.Linear(64, ng),
            nn.Softmax(1),
        )
        self.mean_net = nn.Sequential(
            nn.Linear(288, 64),
            LinearResBlock(64, 64),
            LinearResBlock(64, 64),
            nn.Linear(64, ng * 2),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(288, 64),
            LinearResBlock(64, 64),
            LinearResBlock(64, 64),
            nn.Linear(64, ng * 2),
            nn.ReLU(),
        )
        self.delta_net._modules["3"].bias.data.fill_(10)

        quant_step = stride / 2
        self.loss = GMM2D(ng, quant_step, quant=True, noise=True)
        self.pdist = Gmm2dTable(ng, 601, 3001, stride, stride, device=gid)
        self.sampler = InvTransSampleThreshold(nsample, stride, stride, threshold=1e-10, device=gid)
        self.data_manager = DataManager(nsample, 30, device=gid)
        self.nvigator = PIDNavigator(10, 1, 1, 0.1)
        self.mod, self.pid = npred, 0
        self.last_dim, self.old_n = -1, -1

    def forward(self, video_windows, path_windows, labels):
        nb, bb, bn, c, h, w = video_windows.shape
        n = nb * bb
        vdx = video_windows.view(n * bn, c, h, w)
        ptx = path_windows.view(n, bn, -1, 2)
        lbx = labels.view(n, self.npred, 2)

        br1 = self.vd_net(vdx)
        br1 = self.conv1(br1)
        br1 = br1.view(n, bn, 16, -1)
        br1 = self.conv2(br1)
        br1 = self.vd_pred_net(br1.view(n * self.npred, -1)).view(n, self.npred, -1)
        br2 = self.pt_net(ptx)
        br3 = self.ctx_net(ptx[:, -1, : self.npred], lbx)

        y = torch.cat([br1, br2, br3], dim=2).view(n * self.npred, -1)
        wt = self.weight_net(y)
        mean = self.mean_net(y)
        std = self.delta_net(y) + 1e-6
        lb = lbx.view(n * self.npred, 2)
        return self.loss(mean.view(-1, self.ng, 2), std.view(-1, self.ng, 2), wt, lb)

    def prepare(self, ptx, dim_last):
        changed = self.data_manager.check(ptx)
        if not changed:
            return
        n = ptx.shape[0]
        device = ptx.device
        if self.last_dim != dim_last or self.old_n != n:
            self.y = torch.zeros((n, self.npred, dim_last), dtype=torch.float32, device=device)
            self.tmp_path = torch.zeros((n, self.npred + 1, 2), dtype=torch.float32, device=device)
            self.history = torch.arange(0, n, device=device, dtype=torch.float32)
        if self.y.device != device:
            self.y = self.y.to(device)
            self.tmp_path = self.tmp_path.to(device)
            self.history = self.history.to(device)
        self.old_n, self.last_dim = n, dim_last

    def forward_base(self, video_windows, path_windows, prob):
        n, bn, c, h, w = video_windows.shape
        vdx = video_windows.view(n * bn, c, h, w)
        ptx = path_windows.view(n, bn, -1, 2)

        br1 = self.vd_net(vdx)
        br1 = self.conv1(br1)
        br1 = br1.view(n, bn, 16, -1)
        br1 = self.conv2(br1)
        br1 = self.vd_pred_net(br1.view(n * self.npred, -1)).view(n, self.npred, -1)
        br2 = self.pt_net(ptx)
        self.pbase = br1.shape[-1] + br2.shape[-1]
        br3 = self.ctx_net.forward_base(ptx[:, -1, : self.npred])

        self.prepare(ptx, self.pbase + br3.shape[-1])
        self.y[:, :, : br1.shape[-1]] = br1
        self.y[:, :, br1.shape[-1] : self.pbase] = br2
        self.y[:, :, self.pbase :] = br3
        self.prob = prob
        self.lbx = None
        self.path_stack = {}

    def forward_pred(self):
        n = self.tmp_path.shape[0]
        br3 = self.ctx_net.forward_pred(self.lbx)
        self.y[:, self.pid, self.pbase :] = br3[:, self.pid]
        sy = self.y[:, self.pid]
        wt = self.weight_net(sy)
        mean = self.mean_net(sy)
        std = self.delta_net(sy) + 1e-6
        pdf = self.pdist(wt, mean.view(n, self.ng, 2), std.view(n, self.ng, 2))
        _, loc_target, _ = self.sampler(pdf, self.prob)
        loc = self.nvigator.pid(loc_target)
        nprob, nloc = self.loss.forward_pred(mean.view(n, self.ng, 2), std.view(n, self.ng, 2), wt, loc)
        self.prob = self.prob - torch.log2(nprob)
        self.tmp_path[:, self.pid] = nloc
        self.lbx = nloc
        self.pid = (self.pid + 1) % self.mod
        return self.prob

    def check_error(self, thr=100):
        self.tmp_path[:, self.npred, :] = self.nvigator.get_loc()
        mean_abs_err = torch.mean(self.nvigator.abs_err).item()
        if mean_abs_err < thr:
            return mean_abs_err
        abs_err = torch.mean(self.nvigator.abs_err, dim=1)
        for idx in range(abs_err.shape[0]):
            self.path_stack.setdefault(idx, []).append([abs_err[idx].item(), self.tmp_path[idx].clone()])
        return mean_abs_err

    def clear_stack(self, thr=100):
        total = 0
        self.decay()
        for idx, candidates in self.path_stack.items():
            if candidates[0][0] > thr:
                candidates = sorted(candidates, key=lambda x: x[0])
            self.tmp_path[idx] = candidates[0][1]
            total += candidates[0][0]
        return total / max(len(self.path_stack), 1)

    def decay(self):
        self.nvigator.Ziegler_Nichols(self.nvigator.old_Ku * 0.8, self.nvigator.old_Pu)


def load_checkpoint(model, checkpoint_path, device, strict=True, allow_unsafe=False):
    state = safe_torch_load(checkpoint_path, map_location=device, allow_unsafe=allow_unsafe)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=strict)
    return model
