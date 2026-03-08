
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, Concat, RepConv, DWConv
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.nn.modules.block import DFL

# --------------------------------------------------------------------------------
# 1. StarNet Block
# --------------------------------------------------------------------------------
class StarBlock(nn.Module):
    """
    StarBlock from StarNet.
    Paper: Rewrite the Stars (CVPR 2024)
    Adapted for SDP-YOLO with ReLU6 and BN after DWConv.
    """
    def __init__(self, c1, c2, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # Ensure c1 == c2 because StarBlock usually maintains dimensions
        # If needed, we could add a projection, but standard block assumes dim.
        assert c1 == c2, f"StarBlock expects c1==c2, but got {c1}!={c2}"
        dim = c1
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU6()
        
        mid_dim = int(dim * mlp_ratio)
        self.f1 = nn.Conv2d(dim, mid_dim, 1, bias=False)
        self.f2 = nn.Conv2d(dim, mid_dim, 1, bias=False)
        self.g = nn.Conv2d(mid_dim, dim, 1, bias=False)
        
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x1 = self.f1(x)
        x2 = self.f2(x)
        x = self.g(x1 * x2)
        return x + shortcut

# --------------------------------------------------------------------------------
# 2. DRBNCSPELAN4 Modules
# --------------------------------------------------------------------------------
class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block.
    Training: Large Kernel Conv + Dilated Small Kernel Convs
    Inference: Reparameterized to Single Conv
    """
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=(1, 2, 3), act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.d = d # dilation rates
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
        if p is None:
            p = k // 2
            
        # Training branches
        self.rbr_dense = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.rbr_1x1 = nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False)
        
        # Dilated branches
        self.dilated_branches = nn.ModuleList()
        for dilation in d:
             # Calculate padding for dilated conv to maintain size
             pad = ((k - 1) * dilation) // 2
             self.dilated_branches.append(
                 nn.Conv2d(c1, c2, k, s, pad, dilation=dilation, groups=g, bias=False)
             )
             
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        # Simplification: we don't implement full reparam logic here for simplicity unless requested.
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        for branch in self.dilated_branches:
            out += branch(x)
            
        return self.act(self.bn(out))

class DRBNCSPELAN4(nn.Module):
    """
    DRBNCSPELAN4 module based on RepNCSPELAN4 from YOLOv9 but with DilatedReparamBlock.
    Modified to accept n (repeats) as the 3rd argument for compatibility with parse_model repeat_modules.
    """
    def __init__(self, c1, c2, n=1, c3=64, c5=1):  # ch_in, ch_out, number, ch_hidden, groups
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        # Formula: c3 (split into 2 halves) + n blocks * (c3 // 2)
        # Total channels = c3 + n * (c3 // 2) = (2 + n) * (c3 // 2)
        self.cv3 = Conv((2 + n) * (c3 // 2), c2, 1, 1)  # output conv
        
        # Using DilatedReparamBlock instead of RepNCSP
        self.layers = nn.ModuleList(
            [DilatedReparamBlock(c3 // 2, c3 // 2, k=3) for _ in range(n)]
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in self.layers)
        return self.cv3(torch.cat(y, 1))

# --------------------------------------------------------------------------------
# 3. EPCD Head (Efficient Partial Convolution Detection Head)
# --------------------------------------------------------------------------------

import torch
import torch.nn as nn
import copy
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.block import C2f, Bottleneck, DFL, Proto


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = self.mlp(self.avg_pool(x))
        mx = self.mlp(self.max_pool(x))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, mx], dim=1)
        return torch.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ECA(nn.Module):
    """Efficient Channel Attention (ECA)"""

    def __init__(self, c1: int, c2: int, k_size: int = 3):
        super().__init__()
        channels = c2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class HPA(nn.Module):
    """Hybrid Pooling Attention (HPA) module."""

    def __init__(self, channels: int, groups: int = 16):
        super().__init__()
        g = min(groups, channels)
        if channels % g != 0:
            g = 1
        self.groups = g
        self.cpg = channels // g
        self.conv = nn.Conv2d(self.cpg * 4, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        cpg = self.cpg
        xg = x.reshape(b * g, cpg, h, w)

        fh_avg = F.adaptive_avg_pool2d(xg, (h, 1)).expand(-1, -1, h, w)
        fw_avg = F.adaptive_avg_pool2d(xg, (1, w)).expand(-1, -1, h, w)
        fh_max = F.adaptive_max_pool2d(xg, (h, 1)).expand(-1, -1, h, w)
        fw_max = F.adaptive_max_pool2d(xg, (1, w)).expand(-1, -1, h, w)

        f = torch.cat([fh_avg, fw_avg, fh_max, fw_max], dim=1)
        a = torch.sigmoid(self.conv(f))
        y = xg * a
        return y.reshape(b, c, h, w)


class C3k2CBAM(C2f):
    """C3k2 variant with CBAM replacing the internal convolutional block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CBAM(self.c) for _ in range(n))


class C3k2HPA(C2f):
    """C3k2 variant with HPA replacing the internal convolutional block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(HPA(self.c, groups=16) for _ in range(n))


class CoordAtt(nn.Module):
    """Coordinate Attention (CA)."""

    def __init__(self, c1: int, c2: int, reduction: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        mip = max(8, c2 // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(c2, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        n, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return x * a_h * a_w


class ASF(nn.Module):
    """Attention Scale Fusion (ASF) block for feature reweighting."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg_pool(x))
        return x * w


class C3k2ASF(C2f):
    """C3k2 variant with ASF replacing the internal convolutional block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ASF(self.c) for _ in range(n))
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.tal import make_anchors

class PConv(nn.Module):
    """
    Partial Convolution (Spatial).
    """
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        else:
            self.forward = self.forward_split_cat

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)

class PConvBlock(nn.Module):
    """
    Block combining 1x1 projection (if needed) and PConv.
    Replacement for standard Conv(c1, c2, 3).
    """
    def __init__(self, c1, c2, k=3, s=1, *args, **kwargs):
        super().__init__()
        # If channel change needed, use 1x1 first
        self.proj = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()
        # Then PConv (spatial mixing)
        self.pconv = PConv(c2, n_div=4)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.proj(x)
        x = self.pconv(x)
        return self.act(self.bn(x))

class EPCDDetect(Detect):
    """
    Efficient Partial Convolution based Detect Head (EPCD).
    Replaces standard Conv with PConvBlock in Detect head.
    Coupled Head architecture: Shared PConv features -> 1x1 Box/Cls heads.
    """
    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch) # Pass all args to Detect init
        
        self.nc = nc
        self.reg_max = reg_max
        
        self.nl = len(ch)
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        c_shared = max(c2, c3) # Shared channel width
        
        # Shared branch: PConvBlock x 2
        self.cv_shared = nn.ModuleList(
            nn.Sequential(PConvBlock(x, c_shared), PConvBlock(c_shared, c_shared)) for x in ch
        )
        
        # Heads: 1x1 Conv for Box and Cls
        self.cv2 = nn.ModuleList(nn.Conv2d(c_shared, 4 * self.reg_max, 1) for _ in ch)
        self.cv3 = nn.ModuleList(nn.Conv2d(c_shared, self.nc, 1) for _ in ch)
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        feats = []
        for i in range(self.nl):
            feats.append(self.cv_shared[i](x[i]))

        bs = x[0].shape[0]
        boxes = torch.cat([self.cv2[i](feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([self.cv3[i](feats[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)

        preds = {"boxes": boxes, "scores": scores, "feats": feats}

        if self.end2end:
            if hasattr(self, "one2one_cv2"):
                box_121 = torch.cat(
                    [self.one2one_cv2[i](feats[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
                )
                cls_121 = torch.cat(
                    [self.one2one_cv3[i](feats[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1
                )
                preds["one2one"] = {"boxes": box_121, "scores": cls_121, "feats": feats}

        if self.training:
            return preds

        y = self._inference(preds)
        return y if self.export else (y, preds)

    def _inference(self, x):
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = dist2bbox(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, x["scores"].sigmoid()), 1)
        return y

    def bias_init(self):
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a.bias.data[:] = 1.0
            b.bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class MSEMAttention(nn.Module):
    """Multi-scale efficient attention (lightweight SE+spatial)."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class MSEMDetect(Detect):
    """Detect head with MSEM attention applied to each feature map before prediction."""

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.msem = nn.ModuleList(MSEMAttention(c) for c in ch)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = [m(f) for m, f in zip(self.msem, x)]
        return super().forward(x)


class CWSConv(nn.Module):
    """Channel-Wise Squeeze Convolution (from p2)."""
    def __init__(self, c1, c2, k=3, s=1, *args, **kwargs):
        super().__init__()
        from ultralytics.nn.modules import Conv
        c_ = c1 // 2  # squeeze channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, k, s)

    def forward(self, x):
        return self.cv2(self.cv1(x))
