
import ast
import contextlib
import torch
import torch.nn as nn

# Import all modules required for parse_model
from ultralytics.nn.modules import *
# Ensure Head modules are available
try:
    from ultralytics.nn.modules.head import Detect, Segment, Pose, OBB, WorldDetect, v10Detect
except ImportError:
    pass # Might be already imported via modules

from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.ops import make_divisible

# Import custom modules
import sdp_modules

 # HPA modules
HPA = sdp_modules.HPA
C3k2HPA = sdp_modules.C3k2HPA

# Add custom modules to globals so they can be found by name
StarBlock = sdp_modules.StarBlock
DRBNCSPELAN4 = sdp_modules.DRBNCSPELAN4
EPCDDetect = sdp_modules.EPCDDetect
CBAM = sdp_modules.CBAM
C3k2CBAM = sdp_modules.C3k2CBAM
CoordAtt = sdp_modules.CoordAtt
ASF = sdp_modules.ASF
C3k2ASF = sdp_modules.C3k2ASF
MSEMDetect = sdp_modules.MSEMDetect
ECA = sdp_modules.ECA
CWSConv = sdp_modules.CWSConv


def parse_model(d, ch, verbose=True):
    """Parse a YOLO model.yaml dictionary into a PyTorch model.
    Modified to support SDP-YOLO modules.
    """
    # Args
    legacy = True
    max_channels = float("inf")
    nc, act, scales, end2end = (d.get(x) for x in ("nc", "activation", "scales", "end2end"))
    reg_max = d.get("reg_max", 16)
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape"))
    scale = d.get("scale")
    if scales:
        if not scale:
            scale = next(iter(scales.keys()))
            LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    
    # Custom modules added here
    base_modules = {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            PSA,
            SCDown,
            C2fCIB,
            A2C2f,
            # SDP-YOLO modules
            StarBlock,
            DRBNCSPELAN4,
            CBAM,
            C3k2CBAM,
            CoordAtt,
            ASF,
            C3k2ASF,
            CWSConv,
    }
    if C3k2HPA is not None:
        base_modules.add(C3k2HPA)
    base_modules = frozenset(base_modules)
    
    repeat_modules = {
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            C3x,
            RepC3,
            C2fPSA,
            C2fCIB,
            C2PSA,
            A2C2f,
            # SDP-YOLO modules
            DRBNCSPELAN4,
            C3k2CBAM,
            C3k2ASF,
            ECA,
    }
    if C3k2HPA is not None:
        repeat_modules.add(C3k2HPA)
    repeat_modules = frozenset(repeat_modules)

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m_name = m
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        if m is None:
            raise KeyError(f"Module '{m_name}' is not registered in sdp_utils globals.")
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        if m in base_modules:
            c1, c2 = ch[f], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            if m is C2fAttn:
                args[1] = make_divisible(min(args[1], max_channels // 2) * width, 8)
                args[2] = int(max(round(min(args[2], max_channels // 2 // 32)) * width, 1) if args[2] > 1 else args[2])

            args = [c1, c2, *args[1:]]
            if m in repeat_modules:
                args.insert(2, n)
                n = 1
            if m in {C3k2, C3k2CBAM, C3k2ASF}:
                legacy = False
                if scale in "mlx":
                    args[3] = True
            if m is A2C2f:
                legacy = False
                if scale in "lx":
                    args.extend((True, 1.2))
            if m is C2fCIB:
                legacy = False
        
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in frozenset({HGStem, HGBlock}):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)
                n = 1
        elif m is ResNetLayer:
            c2 = args[1] if args[3] else args[1] * 4
        elif m is torch.nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        
        # Add EPCDDetect here
        elif m in frozenset(
            {
                Detect,
                WorldDetect,
                YOLOEDetect,
                Segment,
                Segment26,
                YOLOESegment,
                YOLOESegment26,
                Pose,
                Pose26,
                OBB,
                OBB26,
                # SDP-YOLO Head
                EPCDDetect,
                MSEMDetect,
            }
        ):
            args.extend([reg_max, end2end, [ch[x] for x in f]])
            if m is Segment or m is YOLOESegment or m is Segment26 or m is YOLOESegment26:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            if m in {Detect, YOLOEDetect, Segment, Segment26, YOLOESegment, YOLOESegment26, Pose, Pose26, OBB, OBB26, EPCDDetect, MSEMDetect}:
                m.legacy = legacy
        
        elif m is v10Detect:
            args.append([ch[x] for x in f])
        elif m is ImagePoolingAttn:
            args.insert(1, [ch[x] for x in f])
        elif m is RTDETRDecoder:
            args.insert(1, [ch[x] for x in f])
        elif m is CBLinear:
            c2 = args[0]
            c1 = ch[f]
            args = [c1, c2, *args[1:]]
        elif m is CBFuse:
            c2 = ch[f[-1]]
        elif m in frozenset({TorchVision, Index}):
            c2 = args[0]
            c1 = ch[f]
            args = [*args[1:]]
        else:
            c2 = ch[f]

        m_ = torch.nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
        t = str(m)[8:-2].replace("__main__.", "")
        m_.np = sum(x.numel() for x in m_.parameters())
        m_.i, m_.f, m_.type = i, f, t
        
        # Init stride and bias for Detect modules
        if isinstance(m_, (Detect, EPCDDetect)):
            s = 256  # 2x min stride
            m_.inplace = True
            
            # Compute stride
            # We need to construct a dummy input
            # args[0] is usually [ch[x] for x in f], which is a list of channels
            # But parse_model has flattened args.
            # However, m_ is already instantiated.
            # We can run a forward pass with dummy input.
            
            # We need to know the input channels for each input to the Detect layer.
            # m.f contains indices of input layers.
            # ch contains output channels of all previous layers.
            
            # Construct dummy input tensor list
            # Note: m_ expects a list of tensors if it has multiple inputs.
            # Detect inputs are usually from different strides.
            
            # For simplicity, we assume standard behavior:
            # Detect forward expects list of tensors.
            # We create dummy tensors based on input channels.
            
            def _forward(m, x):
                return m(x)
            
            # Gather input channels
            # f is list of indices
            if isinstance(f, int):
                input_channels = [ch[f]]
            else:
                input_channels = [ch[x] for x in f]
            
            # Create dummy inputs
            # Assuming s=256 is the input image size? No, s is just a size large enough.
            # We need valid inputs.
            # Standard parse_model uses:
            # forward = lambda x: m_(x)
            # m_.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            # But that assumes m_ takes a single input x (backbone output?) or the model is sequential?
            # Wait, parse_model builds the *whole* model? No, just layers.
            # Detect layer forward expects a list of features.
            
            dummy_input = [torch.zeros(1, c, s, s) for c in input_channels]
            
            # Run forward to get outputs and compute stride
            # EPCDDetect/Detect forward returns dict with "feats" key (list of tensors)
            
            pred = m_(dummy_input)
            
            if isinstance(pred, dict) and "feats" in pred:
                feats = pred["feats"]
            else:
                # Fallback for older versions or if return type differs
                feats = pred
            
            # Compute stride based on output shape vs input size s
            # Output shape[-2] is grid height.
            # Stride = Input Size / Grid Size
            
            m_.stride = torch.tensor([s / x.shape[-2] for x in feats])
            
            # Now call bias_init
            m_.bias_init()
            
        if verbose:
            LOGGER.info(f"{i:>3}{f!s:>20}{n_:>3}{m_.np:10.0f}  {t:<45}{args!s:<30}")
            # print(f"Layer {i}: {t}, args={args}, c2={c2}, ch[-1]={ch[-1] if ch else 'None'}")
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return torch.nn.Sequential(*layers), sorted(save)
