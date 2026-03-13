#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path('/home/tb206/yolo_agri1/github_paper_package')
FIG_DIR = ROOT / 'results' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR = ROOT / 'SCI1' / 'outputs' / 'multiseed' / 'cotton_real' / 'SCI1_baseline_seed3_20260310_181140'
PROP_DIR = ROOT / 'SCI1' / 'outputs' / 'multiseed' / 'cotton_real' / 'SCI1_drbn_wiou_tlpn_seed3_20260310_181140'


def read_img(p: Path):
    if not p.exists():
        raise FileNotFoundError(f'Missing image: {p}')
    return mpimg.imread(str(p))


def save_dual(left_img: Path, right_img: Path, left_title: str, right_title: str, suptitle: str, out_stem: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    axes[0].imshow(read_img(left_img))
    axes[0].set_title(left_title, fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(read_img(right_img))
    axes[1].set_title(right_title, fontsize=11)
    axes[1].axis('off')

    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(FIG_DIR / f'{out_stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{out_stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def save_grid(paths_left, paths_right, row_titles, left_title, right_title, suptitle, out_stem):
    n = len(row_titles)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3.4 * n), constrained_layout=True)
    for i in range(n):
        axes[i, 0].imshow(read_img(paths_left[i]))
        axes[i, 0].set_title(f'{left_title} | {row_titles[i]}', fontsize=10)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(read_img(paths_right[i]))
        axes[i, 1].set_title(f'{right_title} | {row_titles[i]}', fontsize=10)
        axes[i, 1].axis('off')

    fig.suptitle(suptitle, fontsize=13)
    fig.savefig(FIG_DIR / f'{out_stem}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{out_stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    save_dual(
        BASE_DIR / 'confusion_matrix_normalized.png',
        PROP_DIR / 'confusion_matrix_normalized.png',
        'Y11n-B',
        'DWT (Ours)',
        'Normalized Confusion Matrix (cotton_real, seed=3)',
        'fig_exp_confmat_norm_baseline_vs_proposed_cotton_real',
    )

    save_dual(
        BASE_DIR / 'results.png',
        PROP_DIR / 'results.png',
        'Y11n-B',
        'DWT (Ours)',
        'Training Curves (cotton_real, seed=3)',
        'fig_exp_training_curves_baseline_vs_proposed_cotton_real',
    )

    save_dual(
        BASE_DIR / 'BoxPR_curve.png',
        PROP_DIR / 'BoxPR_curve.png',
        'Y11n-B',
        'DWT (Ours)',
        'PR Curve Comparison (cotton_real, seed=3)',
        'fig_exp_pr_curve_baseline_vs_proposed_cotton_real',
    )

    rows = ['val_batch0_pred', 'val_batch1_pred', 'val_batch2_pred']
    save_grid(
        [BASE_DIR / f'{r}.jpg' for r in rows],
        [PROP_DIR / f'{r}.jpg' for r in rows],
        ['Sample-1', 'Sample-2', 'Sample-3'],
        'Y11n-B',
        'DWT (Ours)',
        'Qualitative Prediction Comparison (cotton_real, seed=3)',
        'fig_exp_qualitative_pred_baseline_vs_proposed_cotton_real',
    )

    manifest = FIG_DIR / 'figure_manifest_additional.md'
    manifest.write_text(
        '\n'.join([
            '# Additional Figures Manifest',
            '',
            '- fig_exp_confmat_norm_baseline_vs_proposed_cotton_real.png/.pdf: normalized confusion matrix comparison.',
            '- fig_exp_training_curves_baseline_vs_proposed_cotton_real.png/.pdf: full training curve comparison.',
            '- fig_exp_pr_curve_baseline_vs_proposed_cotton_real.png/.pdf: PR curve comparison.',
            '- fig_exp_qualitative_pred_baseline_vs_proposed_cotton_real.png/.pdf: qualitative prediction comparison on three validation batches.',
        ]) + '\n',
        encoding='utf-8'
    )
    print('saved additional figures to', FIG_DIR)


if __name__ == '__main__':
    main()
