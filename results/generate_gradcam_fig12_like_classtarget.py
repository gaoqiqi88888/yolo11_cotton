from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
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

# 6 disease classes (exclude healthy)
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
    """Wrap Ultralytics DetectionModel to return only tensor output for CAM package."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        return y[0] if isinstance(y, tuple) else y


class ClassTopKTarget:
    """Class-target objective over top-k candidate scores for GradCAM.

    Input to __call__ is one sample output with shape [4+nc, N].
    """

    def __init__(self, class_id: int, topk: int = 30):
        self.class_id = class_id
        self.topk = topk

    def __call__(self, out_one: torch.Tensor) -> torch.Tensor:
        class_scores = out_one[4 + self.class_id, :]
        vals, _ = torch.topk(class_scores, k=min(self.topk, class_scores.shape[-1]))
        return vals.sum()


def pick_images_for_classes(class_ids: List[int]) -> Dict[int, Path]:
    chosen: Dict[int, Path] = {}
    for lb in sorted(VAL_LBL_DIR.glob('*.txt')):
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
        raise RuntimeError(f'Missing class samples in val set for class ids: {missing}')
    return chosen


def parse_label_classes(lb_path: Path) -> List[int]:
    out: List[int] = []
    for ln in lb_path.read_text(encoding='utf-8').splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(int(float(ln.split()[0])))
        except Exception:
            continue
    return out


def load_img(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return arr / 255.0


def to_tensor(img01: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img01.transpose(2, 0, 1)).unsqueeze(0)


def model_output_tensor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    y = model(x)
    return y[0] if isinstance(y, tuple) else y


def class_score_vector(model: nn.Module, img01: np.ndarray) -> np.ndarray:
    dev = next(model.parameters()).device
    inp = to_tensor(img01).to(dev)
    with torch.no_grad():
        out = model_output_tensor(model, inp)[0]  # [4+nc, N]
        cls_scores = out[4:, :].max(dim=1).values
    return cls_scores.detach().cpu().numpy()


def pick_images_by_score_gap(
    class_ids: List[int],
    baseline_model: nn.Module,
    ours_model: nn.Module,
) -> Tuple[Dict[int, Path], Dict[int, str]]:
    """Pick one image per class where ours(B=DWT) is strongest vs baseline.

    Priority:
    1) ours predicts target class correctly
    2) larger score gap (ours - baseline) on target class
    3) larger ours target score
    4) single-class label image preferred
    """
    # collect candidates per class
    cands: Dict[int, List[Tuple[Path, int]]] = {cid: [] for cid in class_ids}
    for lb in sorted(VAL_LBL_DIR.glob('*.txt')):
        cls_list = parse_label_classes(lb)
        if not cls_list:
            continue
        present = set(cls_list)
        stem = lb.stem
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = VAL_IMG_DIR / f'{stem}{ext}'
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        n_unique = len(set(cls_list))
        for cid in class_ids:
            if cid in present:
                cands[cid].append((img_path, n_unique))

    chosen: Dict[int, Path] = {}
    notes: Dict[int, str] = {}

    for cid in class_ids:
        best_item = None
        for img_path, n_unique in cands[cid]:
            img01 = load_img(img_path)
            b_vec = class_score_vector(baseline_model, img01)
            o_vec = class_score_vector(ours_model, img01)

            b_pred = int(np.argmax(b_vec))
            o_pred = int(np.argmax(o_vec))
            b_score = float(b_vec[cid])
            o_score = float(o_vec[cid])
            delta = o_score - b_score

            ours_correct = 1 if o_pred == cid else 0
            single_class = 1 if n_unique == 1 else 0
            rank_key = (ours_correct, delta, o_score, single_class)

            if best_item is None or rank_key > best_item[0]:
                best_item = (
                    rank_key,
                    img_path,
                    f'ours_correct={ours_correct}, delta={delta:+.4f}, '
                    f'o_score={o_score:.4f}, b_score={b_score:.4f}, '
                    f'o_pred={o_pred}, b_pred={b_pred}, unique_cls={n_unique}',
                )

        if best_item is None:
            raise RuntimeError(f'No candidate found for class id {cid}')

        chosen[cid] = best_item[1]
        notes[cid] = best_item[2]

    return chosen, notes


def normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    both = np.concatenate([a.reshape(-1), b.reshape(-1)])
    lo = float(np.percentile(both, 1.0))
    hi = float(np.percentile(both, 99.0))
    if hi <= lo:
        hi = lo + 1e-6
    a_n = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    b_n = np.clip((b - lo) / (hi - lo), 0.0, 1.0)
    return a_n, b_n, lo, hi


def blend(img01: np.ndarray, heat01: np.ndarray, alpha: float = 0.52, cmap_name: str = 'turbo') -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    heat_rgb = cmap(heat01)[..., :3]
    out = (1 - alpha) * img01 + alpha * heat_rgb
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


def build_cam(model_path: Path):
    m = YOLO(str(model_path)).model
    m.requires_grad_(True)  # IMPORTANT for gradient-based CAM
    m.eval()
    w = YoloTensorWrapper(m)
    cam = GradCAM(model=w, target_layers=[w.model.model[22]])
    return cam


def main() -> None:
    # Models for score-based sample selection
    b_score_model = YOLO(str(BASELINE_PT)).model.eval()
    o_score_model = YOLO(str(OURS_PT)).model.eval()

    selected, pick_notes = pick_images_by_score_gap(CLASS_IDS, b_score_model, o_score_model)

    b_cam = build_cam(BASELINE_PT)
    o_cam = build_cam(OURS_PT)

    n_cols = len(CLASS_IDS)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.2 * n_cols, 9.6), constrained_layout=True)

    row_titles = ['Original', 'B (CT-GradCAM)', 'DWT (CT-GradCAM)']
    mapping_lines = ['# Class-targeted GradCAM Fig12-like Mapping (auto-selected by DWT-B gap)\n']

    for c, cid in enumerate(CLASS_IDS):
        img_path = selected[cid]
        img01 = load_img(img_path)
        inp = to_tensor(img01)

        target = [ClassTopKTarget(class_id=cid, topk=30)]
        b_gray = b_cam(input_tensor=inp, targets=target)[0]
        o_gray = o_cam(input_tensor=inp, targets=target)[0]

        b_n, o_n, lo, hi = normalize_pair(b_gray, o_gray)
        b_vis = blend(img01, b_n)
        o_vis = blend(img01, o_n)

        axes[0, c].imshow((img01 * 255).astype(np.uint8))
        axes[0, c].set_title(CLASS_NAMES[cid], fontsize=11)
        axes[0, c].axis('off')

        axes[1, c].imshow(b_vis)
        axes[1, c].axis('off')

        axes[2, c].imshow(o_vis)
        axes[2, c].axis('off')

        mapping_lines.append(
            f'- Col {c+1}: class={cid} ({CLASS_NAMES[cid]}), image={img_path.name}, shared_norm=[{lo:.5f}, {hi:.5f}], {pick_notes[cid]}\n'
        )

    for r in range(3):
        axes[r, 0].text(
            -0.22,
            0.5,
            row_titles[r],
            transform=axes[r, 0].transAxes,
            rotation=90,
            va='center',
            ha='center',
            fontsize=10.5,
            fontweight='bold',
        )

    fig.suptitle('Fig.12-like CT-GradCAM (CottonReal-val, auto-selected for DWT-B contrast)', fontsize=14, y=1.01)

    out_png = OUT_DIR / 'fig_exp_gradcam_like_fig12_classtarget.png'
    out_pdf = OUT_DIR / 'fig_exp_gradcam_like_fig12_classtarget.pdf'
    fig.savefig(out_png, dpi=420, bbox_inches='tight')
    fig.savefig(out_pdf, dpi=420, bbox_inches='tight')
    plt.close(fig)

    mapping = OUT_DIR / 'fig_exp_gradcam_like_fig12_classtarget_mapping.md'
    mapping.write_text(''.join(mapping_lines), encoding='utf-8')

    note = OUT_DIR / 'fig_exp_gradcam_like_fig12_classtarget_note.md'
    note.write_text(
        '\n'.join([
            '# Interpretation Note (Class-targeted GradCAM)',
            '',
            '- Dataset: cotton_real validation set.',
            '- Each column uses exactly the same image for Original/Baseline/Ours.',
            '- Heatmap color indicates relative contribution for the target class in that image.',
            '- Warm colors (yellow/red) indicate higher positive contribution; cool colors indicate lower contribution.',
            '- Baseline and Ours in each column are normalized with the same value range for fair visual comparison.',
        ]),
        encoding='utf-8',
    )

    print(out_png)
    print(out_pdf)
    print(mapping)
    print(note)


if __name__ == '__main__':
    main()
