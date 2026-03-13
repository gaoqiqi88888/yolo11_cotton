from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/tb206/yolo_agri1/github_paper_package/results')
FIG_DIR = ROOT / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

MULTISEED_STATS = ROOT / 'sci1_multiseed_stats_manual.csv'
MAIN_SEED3 = ROOT / 'main_results_seed3_last.csv'
ABL_BEST = ROOT / 'sci1_ablation_in_domain_5variants_bestckpt.csv'

TRIALS_MAIN_CR = Path('/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/multiseed/cotton_real/summary/trials.csv')
TRIALS_MAIN_CX = Path('/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/multiseed/cotton_xevxs_v1/summary/trials.csv')
TRIALS_ABL_CR = Path('/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/ablation_in_domain/cotton_real/summary/trials.csv')
TRIALS_ABL_CX = Path('/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/ablation_in_domain/cotton_xevxs_v1/summary/trials.csv')

VARIANTS_3 = ['baseline', 'drbn_wiou', 'drbn_wiou_tlpn']
VARIANTS_5 = ['baseline', 'drbn_only', 'wiou_only', 'drbn_wiou', 'drbn_wiou_tlpn']
DATASETS_MAIN = ['cotton_real', 'cotton_xevxs_v1', 'plantdoc']
DATASETS_IN = ['cotton_real', 'cotton_xevxs_v1']

DISPLAY = {
    'baseline': 'B',
    'drbn_only': 'D',
    'wiou_only': 'W',
    'drbn_wiou': 'DW',
    'drbn_wiou_tlpn': 'DWT',
    'cotton_real': 'CottonReal',
    'cotton_xevxs_v1': 'CottonXevxs-v1',
    'plantdoc': 'PlantDoc',
}

COLORS = {
    'baseline': '#4C78A8',
    'drbn_only': '#72B7B2',
    'wiou_only': '#54A24B',
    'drbn_wiou': '#F58518',
    'drbn_wiou_tlpn': '#E45756',
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save_both(fig: plt.Figure, stem: str) -> None:
    png = FIG_DIR / f'{stem}.png'
    pdf = FIG_DIR / f'{stem}.pdf'
    fig.savefig(png, dpi=400, bbox_inches='tight')
    fig.savefig(pdf, dpi=400, bbox_inches='tight')
    plt.close(fig)


def fig_main_multiseed_best(rows: List[Dict[str, str]]) -> None:
    # grouped bars: dataset x 3 variants, metric = best_mAP50_95_mean ± std
    w = 0.24
    x = np.arange(len(DATASETS_MAIN))

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for i, v in enumerate(VARIANTS_3):
        means, stds = [], []
        for ds in DATASETS_MAIN:
            r = next(r for r in rows if r['dataset'] == ds and r['variant'] == v)
            means.append(float(r['best_mAP50_95_mean']))
            stds.append(float(r['best_mAP50_95_std']))
        ax.bar(
            x + (i - 1) * w,
            means,
            yerr=stds,
            width=w,
            label=DISPLAY[v],
            color=COLORS[v],
            capsize=3,
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in DATASETS_MAIN])
    ax.set_ylabel('Best mAP@0.5:0.95 (mean ± std)')
    ax.set_title('Main Comparison on Three Datasets (3 seeds)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    save_both(fig, 'fig_exp_main_best_multiseed_bar')


def fig_main_multiseed_last(rows: List[Dict[str, str]]) -> None:
    w = 0.24
    x = np.arange(len(DATASETS_MAIN))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for i, v in enumerate(VARIANTS_3):
        means, stds = [], []
        for ds in DATASETS_MAIN:
            r = next(r for r in rows if r['dataset'] == ds and r['variant'] == v)
            means.append(float(r['mAP50_95_mean']))
            stds.append(float(r['mAP50_95_std']))
        ax.bar(
            x + (i - 1) * w,
            means,
            yerr=stds,
            width=w,
            label=DISPLAY[v],
            color=COLORS[v],
            capsize=3,
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[d] for d in DATASETS_MAIN])
    ax.set_ylabel('Last-epoch mAP@0.5:0.95 (mean ± std)')
    ax.set_title('Main Comparison (Last-Epoch Metric, 3 seeds)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    save_both(fig, 'fig_exp_main_last_multiseed_bar')


def fig_ablation_in_domain_best(rows: List[Dict[str, str]]) -> None:
    x = np.arange(len(VARIANTS_5))
    w = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))

    for i, ds in enumerate(DATASETS_IN):
        means = [float(next(r for r in rows if r['dataset'] == ds and r['variant'] == v)['best_mAP50_95_mean']) for v in VARIANTS_5]
        stds = [float(next(r for r in rows if r['dataset'] == ds and r['variant'] == v)['best_mAP50_95_std']) for v in VARIANTS_5]
        ax.bar(
            x + (i - 0.5) * w,
            means,
            yerr=stds,
            width=w,
            label=DISPLAY[ds],
            capsize=3,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[v] for v in VARIANTS_5], rotation=15, ha='right')
    ax.set_ylabel('Best mAP@0.5:0.95 (mean ± std)')
    ax.set_title('In-domain Ablation (A0-A4, 3 seeds)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(frameon=False)
    save_both(fig, 'fig_exp_ablation_in_domain_best_bar')


def fig_ablation_delta_heatmap(rows: List[Dict[str, str]]) -> None:
    mat = np.zeros((len(DATASETS_IN), len(VARIANTS_5)))
    for i, ds in enumerate(DATASETS_IN):
        for j, v in enumerate(VARIANTS_5):
            r = next(r for r in rows if r['dataset'] == ds and r['variant'] == v)
            mat[i, j] = float(r['delta_vs_baseline'])

    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(mat, cmap='RdYlGn', aspect='auto')
    ax.set_yticks(np.arange(len(DATASETS_IN)))
    ax.set_yticklabels([DISPLAY[d] for d in DATASETS_IN])
    ax.set_xticks(np.arange(len(VARIANTS_5)))
    ax.set_xticklabels([DISPLAY[v] for v in VARIANTS_5], rotation=20, ha='right')
    ax.set_title('Delta vs Baseline (Best mAP@0.5:0.95)')

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f'{mat[i, j]:+.3f}', ha='center', va='center', color='black', fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Gain over Baseline')
    save_both(fig, 'fig_exp_ablation_delta_heatmap')


def fig_efficiency_scatter(rows: List[Dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    markers = {'cotton_real': 'o', 'cotton_xevxs_v1': 's', 'plantdoc': '^'}

    for r in rows:
        ds = r['dataset']
        v = r['variant']
        x = float(r['Params_M'])
        y = float(r['best_mAP50_95'])
        ax.scatter(x, y, s=70, marker=markers[ds], color=COLORS[v], edgecolors='black', linewidths=0.5)
        ax.text(x + 0.006, y + 0.001, f"{DISPLAY[ds]}-{DISPLAY[v]}", fontsize=7)

    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Best mAP@0.5:0.95 (seed=3)')
    ax.set_title('Accuracy-Efficiency Trade-off')
    ax.grid(linestyle='--', alpha=0.4)
    save_both(fig, 'fig_exp_efficiency_accuracy_scatter')


def collect_seed_values() -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {d: {v: [] for v in VARIANTS_5} for d in DATASETS_IN}

    # baseline, drbn_wiou, drbn_wiou_tlpn from main trials
    for ds, path in [('cotton_real', TRIALS_MAIN_CR), ('cotton_xevxs_v1', TRIALS_MAIN_CX)]:
        rows = read_csv(path)
        for v in ['baseline', 'drbn_wiou', 'drbn_wiou_tlpn']:
            out[ds][v] = [float(r['best_mAP50_95']) for r in rows if r['variant'] == v]

    # drbn_only, wiou_only from ablation trials
    for ds, path in [('cotton_real', TRIALS_ABL_CR), ('cotton_xevxs_v1', TRIALS_ABL_CX)]:
        rows = read_csv(path)
        for v in ['drbn_only', 'wiou_only']:
            out[ds][v] = [float(r['best_mAP50_95']) for r in rows if r['variant'] == v]

    return out


def fig_seed_stability_boxplot() -> None:
    vals = collect_seed_values()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=False)

    for ax, ds in zip(axes, DATASETS_IN):
        data = [vals[ds][v] for v in VARIANTS_5]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[DISPLAY[v] for v in VARIANTS_5], showmeans=True)
        for patch, v in zip(bp['boxes'], VARIANTS_5):
            patch.set_facecolor(COLORS[v])
            patch.set_alpha(0.55)
        ax.set_title(DISPLAY[ds])
        ax.set_ylabel('Best mAP@0.5:0.95')
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle('Seed Stability of Ablation Variants (3 seeds)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_both(fig, 'fig_exp_ablation_seed_stability_boxplot')


def write_manifest() -> None:
    items = [
        ('fig_exp_main_best_multiseed_bar', 'Main comparison on three datasets using best-checkpoint metric (mean±std, n=3).'),
        ('fig_exp_main_last_multiseed_bar', 'Main comparison using last-epoch metric (mean±std, n=3).'),
        ('fig_exp_ablation_in_domain_best_bar', 'In-domain ablation A0-A4 grouped bars with error bars.'),
        ('fig_exp_ablation_delta_heatmap', 'Gain/loss heatmap vs baseline for in-domain ablation.'),
        ('fig_exp_efficiency_accuracy_scatter', 'Accuracy-efficiency trade-off scatter (seed=3).'),
        ('fig_exp_ablation_seed_stability_boxplot', 'Seed-level stability boxplots for A0-A4 on in-domain datasets.'),
    ]

    md = FIG_DIR / 'figure_manifest.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write('# Experiment Figures Manifest\n\n')
        for stem, desc in items:
            f.write(f'- {stem}.png / {stem}.pdf: {desc}\n')


def main() -> None:
    rows_stats = read_csv(MULTISEED_STATS)
    rows_main_seed3 = read_csv(MAIN_SEED3)
    rows_abl_best = read_csv(ABL_BEST)

    fig_main_multiseed_best(rows_stats)
    fig_main_multiseed_last(rows_stats)
    fig_ablation_in_domain_best(rows_abl_best)
    fig_ablation_delta_heatmap(rows_abl_best)
    fig_efficiency_scatter(rows_main_seed3)
    fig_seed_stability_boxplot()
    write_manifest()

    print(f'Figures generated in: {FIG_DIR}')


if __name__ == '__main__':
    main()
