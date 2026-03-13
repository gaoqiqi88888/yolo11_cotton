#!/usr/bin/env python3
"""Run SCI1 comparison across multiple seeds and aggregate results.

This script wraps run_sci1_seed3_compare.py and computes mean/std for each
variant on each dataset.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCI1 multi-seed runner")
    p.add_argument(
        "--datasets",
        type=str,
        default=(
            "cotton_real=/home/tb206/yolo_agri1/github_paper_package/SCI1/configs/data_cotton_real_local.yaml,"
            "cotton_xevxs_v1=/home/tb206/yolo_agri1/github_paper_package/SCI1/configs/data_cotton_xevxs_v1_local.yaml,"
            "plantdoc=/home/tb206/yolo_agri1/github_paper_package/SCI1/configs/data_plantdoc_local.yaml"
        ),
        help="Comma-separated dataset_tag=abs_yaml_path entries.",
    )
    p.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seeds")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="0")
    p.add_argument(
        "--variants",
        type=str,
        default="baseline,drbn_wiou,drbn_wiou_tlpn",
        help=(
            "Comma-separated variants passed to run_sci1_seed3_compare.py. "
            "Example for ablation: baseline,drbn_only,wiou_only,drbn_wiou,drbn_wiou_tlpn"
        ),
    )
    p.add_argument(
        "--project-root",
        type=str,
        default="/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/multiseed",
        help="Root output dir. Per-dataset outputs are created under this root.",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="/home/tb206/yolo_agri1/github_paper_package/results",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing rows")
    return p.parse_args()


def parse_dataset_map(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        if "=" not in item:
            raise ValueError(f"Invalid dataset item: {item}")
        tag, yaml_path = item.split("=", 1)
        out[tag.strip()] = yaml_path.strip()
    return out


def run_one(dataset_tag: str, data_yaml: str, seed: int, args: argparse.Namespace) -> None:
    project = Path(args.project_root) / dataset_tag
    project.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "/home/tb206/yolo_agri1/github_paper_package/SCI1/run_sci1_seed3_compare.py",
        "--data",
        data_yaml,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--project",
        str(project),
        "--variants",
        args.variants,
    ]
    if args.force:
        cmd.append("--force")

    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def aggregate(dataset_tag: str, project_root: Path) -> list[dict[str, str]]:
    trials = project_root / dataset_tag / "trials.csv"
    if not trials.exists():
        raise FileNotFoundError(f"Missing trials file: {trials}")

    grouped: dict[str, list[float]] = defaultdict(list)
    with trials.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            v = row.get("mAP50_95") or row.get("map50_95") or "0"
            grouped[variant].append(float(v))

    rows: list[dict[str, str]] = []
    for variant, vals in grouped.items():
        rows.append(
            {
                "dataset": dataset_tag,
                "variant": variant,
                "n": str(len(vals)),
                "map50_95_mean": f"{mean(vals):.6f}",
                "map50_95_std": f"{(stdev(vals) if len(vals) > 1 else 0.0):.6f}",
            }
        )
    rows.sort(key=lambda x: x["variant"])
    return rows


def write_rows(rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "variant", "n", "map50_95_mean", "map50_95_std"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    datasets = parse_dataset_map(args.datasets)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    for tag, yaml_path in datasets.items():
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")
        for s in seeds:
            run_one(tag, yaml_path, s, args)

    all_rows: list[dict[str, str]] = []
    project_root = Path(args.project_root)
    for tag in datasets.keys():
        all_rows.extend(aggregate(tag, project_root))

    out_csv = Path(args.results_dir) / "sci1_multiseed_stats.csv"
    write_rows(all_rows, out_csv)
    print(f"[DONE] Wrote {out_csv}")


if __name__ == "__main__":
    main()
