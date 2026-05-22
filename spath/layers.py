import torch
from typing import Any, Callable, List, Optional
import torch.nn as nn
from torch import Tensor
from SPath_operator import LinearMask, GMM2D, Gmm2dTable, InvTransSample, DataManager

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation,)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1,
        base_width: int = 64,  dilation: int = 1,  norm_layer: Optional[Callable[..., nn.Module]] = None,) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:  identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block: Bottleneck,  layers: List[int], num_classes: int = 1000,  zero_init_residual: bool = False,
        groups: int = 1, width_per_group: int = 64,  replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,) -> None:
        super().__init__()

        if norm_layer is None:   norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes, self.dilation = 64, 1
        if replace_stride_with_dilation is None:  replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups, self.base_width = groups, width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).type(torch.float32).view(1,3,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).type(torch.float32).view(1,3,1,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Bottleneck, planes: int, blocks: int, stride: int = 1, dilate: bool = False,) -> nn.Sequential:
        norm_layer, downsample, previous_dilation = self._norm_layer, None, self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,))
        return nn.Sequential(*layers)

    def cast_input(self,x):
        if not self.mean.device == x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = (x - self.mean) / self.std
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x

    def forward(self, x: Tensor) -> Tensor:
        self.cast_input(x)
        return self._forward_impl(x)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

class LinearResBlock(nn.Module):

    def __init__(self, in_feature, inner_feature, dropout=0.1):
        super(LinearResBlock,self).__init__()
        self.ln1 = nn.Linear(in_feature,inner_feature)
        self.relu1 = nn.PReLU(inner_feature)
        self.ln2 = nn.Linear(inner_feature,in_feature)
        self.bn = nn.LayerNorm(inner_feature)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,x):
        x = x.clone()
        y = self.dropout1(self.relu1(self.ln1(x)))
        y = self.dropout2(self.ln2(y)) + x
        return self.bn(y)

class ConvResBlock(nn.Module):

    def __init__(self, in_feature):
        super(ConvResBlock,self).__init__()
        self.conv1 = nn.Conv1d(in_feature,in_feature,1)
        self.relu1 = nn.PReLU(in_feature)
        self.bn1 = nn.BatchNorm1d(in_feature)
        self.conv2 = nn.Conv1d(in_feature,in_feature,1)
        self.bn2 = nn.BatchNorm1d(in_feature)
        self.relu2 = nn.PReLU(in_feature)

    def forward(self,x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu2(y+x)

class PathNet(nn.Module):

    def __init__(self, nwindows, npred, nfeature=128):
        super(PathNet,self).__init__()
        self.nfeature = nfeature
        self.net1 = nn.Sequential(
            nn.Linear((2*npred+1)*2,nfeature),
            nn.LayerNorm(nfeature),
            LinearResBlock(nfeature,nfeature),
            LinearResBlock(nfeature,nfeature),
            LinearResBlock(nfeature,nfeature),
            LinearResBlock(nfeature,nfeature)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(nwindows,64,1),
            nn.PReLU(64),
            ConvResBlock(64),
            ConvResBlock(64),
            ConvResBlock(64),
            ConvResBlock(64),
            nn.Conv1d(64,npred,1)
        )

    def forward(self,x):
        n,b,w,wb = x.shape
        x = x.view(n*b,w*wb)
        y = self.net1(x)
        y = y.view(n,b,self.nfeature)
        z = self.net2(y)
        return z

class MPReLU(nn.Module):
    def __init__(self, cpg, npred, nctx):
        super(MPReLU,self).__init__()
        ngroup = nctx + npred
        self.weight = nn.Parameter(torch.zeros((cpg*ngroup),dtype=torch.float32)+0.25)
        self.pid, self.mod, self.ctx, self.cpg = 0, npred, nctx, cpg
        self.old_shape = None

    def forward(self,x):
        return nn.functional.prelu(x,self.weight)

    def prepare(self,x):

        self.y = torch.zeros_like(x).to(x.device)
        self.old_shape = x.shape
        return self.y

    def forward_pred(self,x):
        if self.pid == 0:
            wstart, wend = 0, (self.ctx+1)*self.cpg
        else:
            wstart, wend = (self.ctx+self.pid)*self.cpg, (self.ctx+self.pid+1)*self.cpg
        x = x[:,wstart:wend]
        ty = nn.functional.prelu(x,self.weight[wstart:wend])
        self.y[:,wstart:wend] = ty
        self.pid = (self.pid + 1) % self.mod
        return self.y

class MAdd(nn.Module):
    def __init__(self, cpg, npred, nctx):
        super(MAdd,self).__init__()
        self.pid, self.mod, self.ctx, self.cpg = 0, npred, nctx, cpg
        self.old_shape = None

    def forward(self,x,y):
        return x+y

    def prepare(self,x):

        self.z = torch.zeros_like(x).to(x.device)
        self.old_shape = x.shape
        return self.z

    def forward_pred(self,x,y):
        if self.pid == 0:
            wstart, wend = 0, (self.ctx+1)*self.cpg
        else:
            wstart, wend = (self.ctx+self.pid)*self.cpg, (self.ctx+self.pid+1)*self.cpg
        self.z[:,wstart:wend] = x[:,wstart:wend] + y[:,wstart:wend]
        self.pid = (self.pid + 1) % self.mod
        return self.z

class MBNorm(nn.Module):

    def __init__(self, cpg, npred, nctx):
        super(MBNorm,self).__init__()
        self.ngroup = nctx + npred
        self.bn = nn.LayerNorm(cpg)
        self.pid, self.mod, self.ctx, self.cpg = 0, npred, nctx, cpg
        self.old_shape = None

    def forward(self,x):
        n,w = x.shape
        y = self.bn(x.view(n,self.ngroup,-1))
        return y.view(n,w)

    def prepare(self,x):

        self.y = torch.zeros_like(x).to(x.device)
        self.old_shape = x.shape
        return self.y

    def forward_pred(self,x):
        n,w = x.shape
        if self.pid == 0:
            wstart, wend = 0, self.ctx+1
        else:
            wstart, wend = self.ctx+self.pid, self.ctx+self.pid+1
        tx = x[:,wstart*self.cpg:wend*self.cpg].view(n,-1,self.cpg)
        tx = tx.clone()
        ty = self.bn(tx)
        self.y = self.y.clone()
        self.y[:,wstart*self.cpg:wend*self.cpg] = ty.view(n,-1)
        self.pid = (self.pid + 1) % self.mod
        return self.y

class LinearMaskResBlock(nn.Module):

    def __init__(self, cpg, npred, nctx, gid = 0, dropout=0.1):
        super(LinearMaskResBlock,self).__init__()
        nc = cpg*(npred+nctx)
        self.ln1 = LinearMask(nctx,npred,cpg,cpg,device=gid)
        self.relu1 = MPReLU(cpg,npred,nctx)
        self.ln2 = LinearMask(nctx,npred,cpg,cpg,device=gid)
        self.bn = MBNorm(cpg,npred,nctx)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.add = MAdd(cpg,npred,nctx)

    def forward(self,x):
        y = self.dropout1(self.relu1(self.ln1(x)))
        y = self.add(x,self.dropout2(self.ln2(y)))
        return self.bn(y)

    def prepare(self,x):
        y = self.ln1.prepare(x)
        y = self.relu1.prepare(y)
        y = self.ln2.prepare(y)
        y = self.add.prepare(y)
        return self.bn.prepare(y)

    def forward_pred(self,x):
        y = self.ln1.forward_pred(x)
        y = y.clone()
        y = self.relu1.forward_pred(y)
        y = y.clone()
        y = self.ln2.forward_pred(y)
        y = self.add.forward_pred(x,y)
        return self.bn.forward_pred(y)


class ContextNet(nn.Module):

    def __init__(self, npred, nctx, cpg, gid=0):
        super(ContextNet,self).__init__()
        self.cpg = cpg
        self.bt1 = nn.Linear(npred*2,nctx*cpg)
        self.bt2 = nn.Linear(2,cpg)
        self.net = nn.Sequential(
            LinearMask(nctx,npred,cpg,cpg,False,device=gid),
            MPReLU(cpg,npred,nctx),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMask(nctx,npred,cpg,cpg,True,True,device=gid)
        )
        self.mod, self.nctx, self.pid = npred, nctx, 0
        self.pbase = nctx*cpg
        self.old_shape = None
        self.tl = None

    def forward(self,x,y):
        n,w,b = x.shape
        x = x.view(n,w*b)
        y = y.view(-1,b)
        tx = self.bt1(x)
        ty = self.bt2(y).view(n,w*self.cpg)
        tt = torch.cat([tx,ty],dim=1)
        z = self.net(tt)
        return z.view(n,w,self.cpg)

    def prepare(self,x):
        n,w,_ = x.shape
        self.tt = torch.zeros((n,self.pbase+w*self.cpg),dtype=torch.float32,device=x.device)
        modified = True
        if modified:
            tl = self.tt
            for layer in self.net:
                tl = layer.prepare(tl)
            self.tlv = tl

        self.old_shape = x.shape
        return self.tlv

    def forward_base(self,x):
        tl = self.prepare(x)
        n,w,b = x.shape
        x = x.view(n,w*b)
        tx = self.bt1(x)
        self.tt[:,:self.pbase] = tx
        self.tt[:,self.pbase:] = 0
        return tl.view(n,-1,self.cpg)


    def forward_pred(self,yp):
        n = self.tt.shape[0]
        if self.pid > 0:
            yp = yp.view(n,-1)
            ypt = self.bt2(yp).view(n,self.cpg)
            pb = self.pbase+(self.pid-1)*self.cpg
            self.tt = self.tt.clone()
            self.tt[:,pb:pb+self.cpg] = ypt
        tl = self.tt
        for layer in self.net:
            tl = tl.clone()
            tl = layer.forward_pred(tl)
        self.pid = (self.pid + 1) % self.mod
        return tl.view(n,-1,self.cpg)

class ContextNetV2(nn.Module):

    def __init__(self, npred,  cpg, gid=0):
        super(ContextNetV2,self).__init__()
        nctx = 0
        self.cpg = cpg
        self.npred = npred
        self.bt2 = nn.Linear(2,cpg)
        self.net = nn.Sequential(
            LinearMask(nctx,npred,cpg,cpg,False,device=gid),
            MPReLU(cpg,npred,nctx),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMaskResBlock(cpg,npred,nctx,gid),
            LinearMask(nctx,npred,cpg,cpg,True,True,device=gid)
        )
        self.mod, self.pid = npred, 0
        self.old_shape = -1
        self.tl = None

    def forward(self,y):
        n,w,b = y.shape
        y = y.view(-1,b)
        tt = self.bt2(y).view(n,w*self.cpg)
        z = self.net(tt)
        return z.view(n,w,self.cpg)

    def prepare(self,x):
        num = x.shape[0]
        self.tt = torch.zeros((num,self.npred*self.cpg),dtype=torch.float32,device=x.device)
        modified = True
        if modified:
            tl = self.tt
            for layer in self.net:
                tl = layer.prepare(tl)
            self.tlv = tl

        self.old_shape = num
        return self.tlv

    def forward_base(self,x):
        tl = self.prepare(x)
        self.tt[:,:] = 0
        n = x.shape[0]
        return tl.view(n,-1,self.cpg)


    def forward_pred(self,yp):
        n = self.tt.shape[0]
        if self.pid > 0:
            yp = yp.view(n,-1)
            ypt = self.bt2(yp).view(n,self.cpg)
            pb = (self.pid-1)*self.cpg
            self.tt = self.tt.clone()
            self.tt[:,pb:pb+self.cpg] = ypt
        tl = self.tt
        for layer in self.net:
            tl = tl.clone()
            tl = layer.forward_pred(tl)
        self.pid = (self.pid + 1) % self.mod
        return tl.view(n,-1,self.cpg)

@torch.no_grad()
def manange_data(m, dm):
    tm = type(m)
    if tm == MBNorm:
        dm.push(m.y)
    elif tm == MAdd:
        dm.push(m.z)
    elif tm == MPReLU:
        dm.push(m.y)
    elif tm == LinearMask:
        dm.push(m.out)
