from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

# ---------------- Paths ----------------
ROOT = Path('/home/tb206/yolo_agri1')
BACKUP_ROOT = ROOT / 'github_paper_package/backupyolo'
if str(BACKUP_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKUP_ROOT))

# custom modules for checkpoint loading
import sdp_modules  # noqa: F401,E402
import sdp_loss  # noqa: F401,E402
import sdp_utils  # noqa: F401,E402

tasks.parse_model = sdp_utils.parse_model

DATA_ROOT = ROOT / 'datasets/cotton_xevxs_v1'
IMG_DIR = DATA_ROOT / 'valid/images'
LBL_DIR = DATA_ROOT / 'valid/labels'

OUT_DIR = ROOT / 'github_paper_package/results/figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PT = ROOT / 'github_paper_package/SCI1/outputs/multiseed/cotton_xevxs_v1/SCI1_baseline_seed3_20260310_222412/weights/best.pt'
PROPOSED_PT = ROOT / 'github_paper_package/SCI1/outputs/multiseed/cotton_xevxs_v1/SCI1_drbn_wiou_seed3_20260310_222412/weights/best.pt'

# xevxs classes: 0 blight, 1 curl, 2 grey mildew, 3 healthy, 4 leaf spot, 5 wilt
DISEASE_CLASS_IDS = [0, 1, 2, 4, 5]  # exclude healthy
CLASS_NAMES = {
    0: 'blight',
    1: 'curl',
    2: 'grey mildew',
    3: 'healthy',
    4: 'leaf spot',
    5: 'wilt',
}

IMG_SIZE = 640
N_VIS_PER_CLASS = 1
N_METRIC_PER_CLASS = 20
TOPQ = 0.90  # top-10% pixels


class YoloTensorWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y[0] if isinstance(y, tuple) else y


class ClassTopKTarget:
    def __init__(self, class_id: int, topk: int = 30):
        self.class_id = class_id
        self.topk = topk

    def __call__(self, out_one: torch.Tensor) -> torch.Tensor:
        cls_scores = out_one[4 + self.class_id, :]
        vals, _ = torch.topk(cls_scores, k=min(self.topk, cls_scores.shape[-1]))
        return vals.sum()


def load_img(path: Path, size: int = IMG_SIZE) -> np.ndarray:
    arr = np.array(Image.open(path).convert('RGB').resize((size, size)), dtype=np.float32)
    return arr / 255.0


def to_tensor(img01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img01.transpose(2, 0, 1)).unsqueeze(0)


def parse_boxes(label_path: Path, class_id: int, size: int = IMG_SIZE) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    for ln in label_path.read_text(encoding='utf-8').splitlines():
        sp = ln.strip().split()
        if len(sp) != 5:
            continue
        cid = int(float(sp[0]))
        if cid != class_id:
            continue
        xc, yc, w, h = map(float, sp[1:])
        x1 = max(0, int((xc - w / 2) * size))
        y1 = max(0, int((yc - h / 2) * size))
        x2 = min(size, int((xc + w / 2) * size))
        y2 = min(size, int((yc + h / 2) * size))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    s = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.percentile(s, 1.0))
    hi = float(np.percentile(s, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    a_n = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    b_n = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    return a_n, b_n


def blend(img01: np.ndarray, heat01: np.ndarray, alpha: float = 0.52, cmap_name: str = 'turbo') -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    heat_rgb = cmap(heat01)[..., :3]
    out = (1 - alpha) * img01 + alpha * heat_rgb
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)


def build_cam(weight: Path) -> GradCAM:
    m = YOLO(str(weight)).model
    m.requires_grad_(True)
    m.eval()
    w = YoloTensorWrapper(m)
    return GradCAM(model=w, target_layers=[w.model.model[22]])


def collect_class_files(class_id: int) -> List[Tuple[Path, Path]]:
    out = []
    for lb in sorted(LBL_DIR.glob('*.txt')):
        lines = [ln.strip() for ln in lb.read_text(encoding='utf-8').splitlines() if ln.strip()]
        has_cls = False
        for ln in lines:
            sp = ln.split()
            if len(sp) != 5:
                continue
            if int(float(sp[0])) == class_id:
                has_cls = True
                break
        if not has_cls:
            continue
        stem = lb.stem
        img = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = IMG_DIR / f'{stem}{ext}'
            if p.exists():
                img = p
                break
        if img is not None:
            out.append((img, lb))
    return out


def cam_metrics(gray: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    mask = np.zeros_like(gray, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = True

    if mask.sum() == 0 or (~mask).sum() == 0:
        return float('nan'), float('nan')

    gray_pos = np.maximum(gray, 0.0)
    mass_in = float(gray_pos[mask].sum() / (gray_pos.sum() + 1e-8))

    th = float(np.quantile(gray, TOPQ))
    top_mask = gray >= th
    hit = float((top_mask & mask).sum() / (top_mask.sum() + 1e-8))
    return mass_in, hit


def mean_std(xs: List[float]) -> Tuple[float, float, int]:
    xs = [x for x in xs if np.isfinite(x)]
    if not xs:
        return float('nan'), float('nan'), 0
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) >= 2 else 0.0
    return m, s, len(xs)


def main() -> None:
    base_cam = build_cam(BASELINE_PT)
    prop_cam = build_cam(PROPOSED_PT)

    # Visualization panel data (1 sample per disease class)
    vis_samples: List[Tuple[int, Path, Path]] = []
    for cid in DISEASE_CLASS_IDS:
        files = collect_class_files(cid)
        vis_samples.extend([(cid, files[i][0], files[i][1]) for i in range(min(N_VIS_PER_CLASS, len(files)))])

    # Generate panel: Original / Baseline / Proposed
    n_cols = len(vis_samples)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.0 * n_cols, 9.2), constrained_layout=True)
    if n_cols == 1:
        axes = np.array(axes).reshape(3, 1)

    row_titles = ['Original', 'B (CT-GradCAM)', 'DW (CT-GradCAM)']
    mapping_lines = ['# Cotton_xevxs_v1 class-targeted CAM mapping\n']

    for c, (cid, img_path, lb_path) in enumerate(vis_samples):
        img01 = load_img(img_path)
        inp = to_tensor(img01)
        target = [ClassTopKTarget(class_id=cid, topk=30)]

        b_gray = base_cam(input_tensor=inp, targets=target)[0]
        p_gray = prop_cam(input_tensor=inp, targets=target)[0]
        b_n, p_n = normalize_pair(b_gray, p_gray)

        axes[0, c].imshow((img01 * 255).astype(np.uint8))
        axes[0, c].set_title(CLASS_NAMES[cid], fontsize=11)
        axes[0, c].axis('off')

        axes[1, c].imshow(blend(img01, b_n))
        axes[1, c].axis('off')

        axes[2, c].imshow(blend(img01, p_n))
        axes[2, c].axis('off')

        mapping_lines.append(f'- Col {c+1}: class={cid} ({CLASS_NAMES[cid]}), image={img_path.name}\n')

    for r in range(3):
        axes[r, 0].text(
            -0.23,
            0.5,
            row_titles[r],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            fontweight='bold',
        )

    fig.suptitle('Class-targeted GradCAM on cotton_xevxs_v1 (disease-level localization)', fontsize=14, y=1.01)
    panel_png = OUT_DIR / 'fig_exp_gradcam_xevxs_classtarget.png'
    panel_pdf = OUT_DIR / 'fig_exp_gradcam_xevxs_classtarget.pdf'
    fig.savefig(panel_png, dpi=420, bbox_inches='tight')
    fig.savefig(panel_pdf, dpi=420, bbox_inches='tight')
    plt.close(fig)

    (OUT_DIR / 'fig_exp_gradcam_xevxs_classtarget_mapping.md').write_text(''.join(mapping_lines), encoding='utf-8')

    # Quantitative CAM localization metrics
    raw_rows = []
    summary_rows = []

    for cid in DISEASE_CLASS_IDS:
        files = collect_class_files(cid)[:N_METRIC_PER_CLASS]
        br, bh, pr, ph = [], [], [], []

        for img_path, lb_path in files:
            img01 = load_img(img_path)
            inp = to_tensor(img01)
            target = [ClassTopKTarget(class_id=cid, topk=30)]

            b_gray = base_cam(input_tensor=inp, targets=target)[0]
            p_gray = prop_cam(input_tensor=inp, targets=target)[0]
            boxes = parse_boxes(lb_path, class_id=cid)

            b_ratio, b_hit = cam_metrics(b_gray, boxes)
            p_ratio, p_hit = cam_metrics(p_gray, boxes)

            br.append(b_ratio)
            bh.append(b_hit)
            pr.append(p_ratio)
            ph.append(p_hit)

            raw_rows.append(
                {
                    'class_id': cid,
                    'class_name': CLASS_NAMES[cid],
                    'image': img_path.name,
                    'baseline_cam_mass_in': b_ratio,
                    'baseline_hit_top10': b_hit,
                    'proposed_cam_mass_in': p_ratio,
                    'proposed_hit_top10': p_hit,
                }
            )

        bm, bs, bn = mean_std(br)
        bhm, bhs, bhn = mean_std(bh)
        pm, ps, pn = mean_std(pr)
        phm, phs, phn = mean_std(ph)

        summary_rows.append(
            {
                'class_id': cid,
                'class_name': CLASS_NAMES[cid],
                'n_images': min(bn, pn),
                'baseline_mass_in_mean': bm,
                'baseline_mass_in_std': bs,
                'proposed_mass_in_mean': pm,
                'proposed_mass_in_std': ps,
                'delta_mass_in': pm - bm,
                'baseline_hit_top10_mean': bhm,
                'baseline_hit_top10_std': bhs,
                'proposed_hit_top10_mean': phm,
                'proposed_hit_top10_std': phs,
                'delta_hit_top10': phm - bhm,
            }
        )

    raw_csv = OUT_DIR / 'cam_localization_xevxs_raw.csv'
    with open(raw_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'class_id',
                'class_name',
                'image',
                'baseline_cam_mass_in',
                'baseline_hit_top10',
                'proposed_cam_mass_in',
                'proposed_hit_top10',
            ],
        )
        w.writeheader()
        w.writerows(raw_rows)

    summary_csv = OUT_DIR / 'cam_localization_xevxs_summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                'class_id',
                'class_name',
                'n_images',
                'baseline_mass_in_mean',
                'baseline_mass_in_std',
                'proposed_mass_in_mean',
                'proposed_mass_in_std',
                'delta_mass_in',
                'baseline_hit_top10_mean',
                'baseline_hit_top10_std',
                'proposed_hit_top10_mean',
                'proposed_hit_top10_std',
                'delta_hit_top10',
            ],
        )
        w.writeheader()
        w.writerows(summary_rows)

    # macro row
    val_cols = [
        'baseline_mass_in_mean',
        'proposed_mass_in_mean',
        'delta_mass_in',
        'baseline_hit_top10_mean',
        'proposed_hit_top10_mean',
        'delta_hit_top10',
    ]
    macro = {k: float(np.mean([float(r[k]) for r in summary_rows])) for k in val_cols}

    macro_md = OUT_DIR / 'cam_localization_xevxs_macro.md'
    macro_md.write_text(
        '\n'.join(
            [
                '# CAM localization macro summary (cotton_xevxs_v1)',
                f"- baseline_mass_in_mean: {macro['baseline_mass_in_mean']:.4f}",
                f"- proposed_mass_in_mean: {macro['proposed_mass_in_mean']:.4f}",
                f"- delta_mass_in: {macro['delta_mass_in']:+.4f}",
                f"- baseline_hit_top10_mean: {macro['baseline_hit_top10_mean']:.4f}",
                f"- proposed_hit_top10_mean: {macro['proposed_hit_top10_mean']:.4f}",
                f"- delta_hit_top10: {macro['delta_hit_top10']:+.4f}",
            ]
        ),
        encoding='utf-8',
    )

    print(panel_png)
    print(panel_pdf)
    print(raw_csv)
    print(summary_csv)
    print(macro_md)


if __name__ == '__main__':
    main()
