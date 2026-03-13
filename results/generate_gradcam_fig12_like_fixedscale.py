from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import EigenCAM
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
        return y[0] if isinstance(y, tuple) else y


def pick_images_for_classes(class_ids: List[int]) -> Dict[int, Path]:
    chosen: Dict[int, Path] = {}
    label_files = sorted(VAL_LBL_DIR.glob('*.txt'))

    for lb in label_files:
        lines = [ln.strip() for ln in lb.read_text(encoding='utf-8').splitlines() if ln.strip()]
        present = set()
        for ln in lines:
            try:
                present.add(int(float(ln.split()[0])))
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
        raise RuntimeError(f'Missing classes in selected images: {missing}')
    return chosen


def load_img(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return arr / 255.0


def to_tensor(img01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img01.transpose(2, 0, 1)).unsqueeze(0)


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    # shared normalization for fair visual comparison in one column
    stack = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.percentile(stack, 1.0))
    hi = float(np.percentile(stack, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    a_n = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    b_n = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    return a_n, b_n, lo, hi


def blend_with_cmap(img01: np.ndarray, heat01: np.ndarray, alpha: float = 0.5, cmap_name: str = 'turbo') -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    heat_rgb = cmap(heat01)[..., :3]
    out = (1 - alpha) * img01 + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def main() -> None:
    selected = pick_images_for_classes(CLASS_IDS)

    baseline = YOLO(str(BASELINE_PT)).model
    ours = YOLO(str(OURS_PT)).model

    b_wrap = YoloTensorWrapper(baseline)
    o_wrap = YoloTensorWrapper(ours)

    b_cam = EigenCAM(model=b_wrap, target_layers=[b_wrap.model.model[22]])
    o_cam = EigenCAM(model=o_wrap, target_layers=[o_wrap.model.model[22]])

    n_cols = len(CLASS_IDS)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.4), constrained_layout=True)

    row_titles = ['Original', 'YOLO11n-Baseline (EigenCAM)', 'Proposed (DRBN+WIoU+TLPN, EigenCAM)']

    mapping_lines = ['# Column-to-image mapping for Fig12-like heatmap\n']

    for c, cid in enumerate(CLASS_IDS):
        img_path = selected[cid]
        img01 = load_img(img_path)

        b_gray = b_cam(input_tensor=to_tensor(img01))[0]
        o_gray = o_cam(input_tensor=to_tensor(img01))[0]

        # IMPORTANT: shared scaling per column (baseline vs ours)
        b_n, o_n, lo, hi = normalize_pair(b_gray, o_gray)

        b_vis = blend_with_cmap(img01, b_n, alpha=0.52, cmap_name='turbo')
        o_vis = blend_with_cmap(img01, o_n, alpha=0.52, cmap_name='turbo')

        axes[0, c].imshow((img01 * 255).astype(np.uint8))
        axes[0, c].set_title(CLASS_NAMES[cid], fontsize=11)
        axes[0, c].axis('off')

        axes[1, c].imshow(b_vis)
        axes[1, c].axis('off')

        axes[2, c].imshow(o_vis)
        axes[2, c].axis('off')

        mapping_lines.append(
            f'- Col {c+1}: class={cid} ({CLASS_NAMES[cid]}), image={img_path.name}, shared_norm=[{lo:.5f}, {hi:.5f}]\n'
        )

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

    fig.suptitle('Fig.12-like Heatmap Visualization (Shared Color Scale per Sample)', fontsize=14, y=1.01)

    out_png = OUT_DIR / 'fig_exp_gradcam_like_fig12_fixedscale.png'
    out_pdf = OUT_DIR / 'fig_exp_gradcam_like_fig12_fixedscale.pdf'
    fig.savefig(out_png, dpi=420, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=420, bbox_inches='tight')
    plt.close(fig)

    mapping_path = OUT_DIR / 'fig_exp_gradcam_like_fig12_fixedscale_mapping.md'
    mapping_path.write_text(''.join(mapping_lines), encoding='utf-8')

    print(out_png)
    print(out_pdf)
    print(mapping_path)


if __name__ == '__main__':
    main()
