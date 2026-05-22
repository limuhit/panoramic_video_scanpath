import torch
import torch.nn as nn
import math

class QUANT_AF(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, noise):
        if noise:
            nt = torch.rand_like(x) - 0.5
            return nt + x
        else:
            x_q = torch.round(x)
            return x_q

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None


class Quant(nn.Module):

    def __init__(self,step=1,noise=False):
        super(Quant,self).__init__()
        self.noise = noise
        self.step_inv = 1. / step
        self.step = step

    def forward(self,x):
        tx = x * self.step_inv
        if self.training:
            y = QUANT_AF.apply(tx,self.noise)
        else:
            y = QUANT_AF.apply(tx,False)
        ty = y * self.step
        return ty


class GMM2D(nn.Module):

    def __init__(self, num_gaussian, quant_step=0.5, quant=False, noise=False):
        super(GMM2D,self).__init__()
        self.num_gaussian = num_gaussian
        self.step = quant_step
        self.s2 = 1. / math.sqrt(2)
        self.quant = quant
        self.quant_op = Quant(quant_step*2,noise) if quant else None

    def forward(self, mean, std, weight, loc):
        """Compute the probability of each target location under a 2D GMM."""
        assert(mean.shape[1]==self.num_gaussian)
        n = mean.shape[0]
        mean = mean.view(n,self.num_gaussian,2)
        std = std.view(n,self.num_gaussian,2)
        weight = weight.view(n,self.num_gaussian)
        if self.quant:  loc = self.quant_op(loc)
        loc = loc.view(n,2).repeat(1,self.num_gaussian).view(n,self.num_gaussian,2)
        xy = (loc - mean) / std * self.s2
        delta = self.step / std * self.s2
        p1 = torch.erf(xy+delta) - torch.erf(xy-delta)
        p2 = p1[:,:,0] * p1[:,:,1] * 0.25
        prob = torch.sum(p2*weight, dim=1)
        return prob

    def forward_pred(self, mean, std, weight, loc):
        with torch.no_grad():
            assert(mean.shape[1]==self.num_gaussian)
            n = mean.shape[0]
            mean = mean.view(n,self.num_gaussian,2)
            std = std.view(n,self.num_gaussian,2)
            weight = weight.view(n,self.num_gaussian)
            loc_q = self.quant_op(loc) if self.quant else loc
            loc = loc_q.view(n,2).repeat(1,self.num_gaussian).view(n,self.num_gaussian,2)
            xy = (loc - mean) / std * self.s2
            delta = self.step / std * self.s2
            p1 = torch.erf(xy+delta) - torch.erf(xy-delta)
            p2 = p1[:,:,0] * p1[:,:,1] * 0.25
            prob = torch.sum(p2*weight, dim=1)
            return prob,loc_q
