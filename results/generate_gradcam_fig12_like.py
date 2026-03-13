from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# ---------------- Paths ----------------
ROOT = Path('/home/tb206/yolo_agri1')
BACKUP_ROOT = ROOT / 'github_paper_package/backupyolo'
if str(BACKUP_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKUP_ROOT))

# required by custom checkpoints
import sdp_modules  # noqa: F401,E402
import sdp_loss  # noqa: F401,E402
import sdp_utils  # noqa: F401,E402
tasks.parse_model = sdp_utils.parse_model

VAL_IMG_DIR = ROOT / 'datasets/cotton_real/images/val'
VAL_LBL_DIR = ROOT / 'datasets/cotton_real/labels/val'
OUT_DIR = ROOT / 'github_paper_package/results/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PT = ROOT / 'github_paper_package/SCI1/outputs/multiseed/cotton_real/SCI1_baseline_seed3_20260310_181140/weights/best.pt'
OURS_PT = ROOT / 'github_paper_package/SCI1/outputs/multiseed/cotton_real/SCI1_drbn_wiou_tlpn_seed3_20260310_181140/weights/best.pt'

# choose 6 disease classes (exclude healthy)
CLASS_IDS = [0, 1, 2, 3, 6, 7]
CLASS_NAMES = {
    0: 'Aphids',
    1: 'Army worm',
    2: 'Bacterial blight',
    3: 'Cotton Boll Rot',
    4: 'Green Cotton Boll',
    5: 'Healthy',
    6: 'Powdery mildew',
    7: 'Target spot',
}

IMG_SIZE = 640


class YoloTensorWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        # ultralytics DetectionModel often returns (pred, aux_dict)
        return y[0] if isinstance(y, tuple) else y


def pick_images_for_classes(class_ids: List[int]) -> Dict[int, Path]:
    chosen: Dict[int, Path] = {}
    label_files = sorted(VAL_LBL_DIR.glob('*.txt'))

    for lb in label_files:
        if not lb.exists():
            continue
        lines = [ln.strip() for ln in lb.read_text(encoding='utf-8').splitlines() if ln.strip()]
        present = set()
        for ln in lines:
            try:
                cid = int(float(ln.split()[0]))
                present.add(cid)
            except Exception:
                continue

        for cid in class_ids:
            if cid in present and cid not in chosen:
                stem = lb.stem
                for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                    p = VAL_IMG_DIR / f'{stem}{ext}'
                    if p.exists():
                        chosen[cid] = p
                        break

        if len(chosen) == len(class_ids):
            break

    missing = [cid for cid in class_ids if cid not in chosen]
    if missing:
        raise RuntimeError(f'Could not find validation images for class ids: {missing}')

    return chosen


def load_img_as_float01(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    arr = np.array(Image.open(path).convert('RGB').resize((size, size)), dtype=np.float32)
    return arr / 255.0


def to_tensor(img01: np.ndarray) -> torch.Tensor:
    # HWC [0,1] -> BCHW
    return torch.from_numpy(img01.transpose(2, 0, 1)).unsqueeze(0)


def cam_overlay(cam: EigenCAM, img01: np.ndarray) -> np.ndarray:
    inp = to_tensor(img01)
    gray = cam(input_tensor=inp)[0]
    overlay = show_cam_on_image(img01, gray, use_rgb=True)
    return overlay


def main() -> None:
    selected = pick_images_for_classes(CLASS_IDS)

    baseline = YOLO(str(BASELINE_PT)).model
    ours = YOLO(str(OURS_PT)).model

    b_wrap = YoloTensorWrapper(baseline)
    o_wrap = YoloTensorWrapper(ours)

    # last neck layer before Detect
    b_cam = EigenCAM(model=b_wrap, target_layers=[b_wrap.model.model[22]])
    o_cam = EigenCAM(model=o_wrap, target_layers=[o_wrap.model.model[22]])

    n_cols = len(CLASS_IDS)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.1 * n_cols, 9.2), constrained_layout=True)

    row_titles = ['Original', 'B (EigenCAM)', 'DWT (EigenCAM)']

    for c, cid in enumerate(CLASS_IDS):
        img_path = selected[cid]
        img01 = load_img_as_float01(img_path)

        b_vis = cam_overlay(b_cam, img01)
        o_vis = cam_overlay(o_cam, img01)

        # Row 1: original
        axes[0, c].imshow((img01 * 255).astype(np.uint8))
        axes[0, c].set_title(CLASS_NAMES[cid], fontsize=11)
        axes[0, c].axis('off')

        # Row 2: baseline CAM
        axes[1, c].imshow(b_vis)
        axes[1, c].axis('off')

        # Row 3: proposed CAM
        axes[2, c].imshow(o_vis)
        axes[2, c].axis('off')

    for r in range(3):
        axes[r, 0].text(
            -0.18,
            0.5,
            row_titles[r],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=11,
            fontweight='bold',
        )

    fig.suptitle('Heatmap Visualization Similar to Fig.12 (Cotton Disease Samples)', fontsize=14, y=1.01)

    out_png = OUT_DIR / 'fig_exp_gradcam_like_fig12.png'
    out_pdf = OUT_DIR / 'fig_exp_gradcam_like_fig12.pdf'
    fig.savefig(out_png, dpi=400, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=400, bbox_inches='tight')
    plt.close(fig)

    print(out_png)
    print(out_pdf)


if __name__ == '__main__':
    main()
