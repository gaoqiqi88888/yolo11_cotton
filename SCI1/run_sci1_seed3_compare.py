import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
BACKUP_ROOT = os.path.join(ROOT, "backupyolo")
if BACKUP_ROOT not in sys.path:
    sys.path.insert(0, BACKUP_ROOT)

import sdp_loss  # noqa: E402
import sdp_utils  # noqa: E402


@dataclass
class Variant:
    name: str
    yaml_path: str
    use_wiou: bool
    sci_name: str


def set_wiou(enabled: bool):
    if enabled:
        sdp_loss.patch_loss()
    else:
        if hasattr(sdp_loss, "restore_loss"):
            sdp_loss.restore_loss()


def ensure_csv(trials_csv: str):
    os.makedirs(os.path.dirname(trials_csv), exist_ok=True)
    if os.path.exists(trials_csv):
        return
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sci_name",
                "seed",
                "epochs",
                "use_wiou",
                "yaml_path",
                "P",
                "R",
                "mAP50",
                "mAP50_95",
                "best_mAP50",
                "best_mAP50_95",
                "best_epoch",
                "Params_M",
                "FLOPs_G",
                "RunTime_s",
                "run_name",
                "save_dir",
                "results_csv",
                "weights",
            ],
        )
        w.writeheader()


def load_done(trials_csv: str):
    if not os.path.exists(trials_csv):
        return set()
    with open(trials_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["variant"], str(r["seed"])) for r in rows}


def append_row(trials_csv: str, row: Dict):
    with open(trials_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writerow(row)


def read_metrics(results_csv: str) -> Dict[str, float]:
    keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    alt = ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"]
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1] if rows else {}

    def pick(i):
        return float(last.get(keys[i]) or last.get(alt[i]) or 0.0)

    return {
        "P": pick(0),
        "R": pick(1),
        "mAP50": pick(2),
        "mAP50_95": pick(3),
    }


def best_metrics(results_csv: str) -> Dict[str, float]:
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"best_mAP50": 0.0, "best_mAP50_95": 0.0, "best_epoch": -1}

    def get(r, k1, k2):
        return float(r.get(k1) or r.get(k2) or 0.0)

    best = max(rows, key=lambda r: get(r, "metrics/mAP50-95(B)", "metrics/mAP50-95"))
    return {
        "best_mAP50": get(best, "metrics/mAP50(B)", "metrics/mAP50"),
        "best_mAP50_95": get(best, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
        "best_epoch": int(float(best.get("epoch", -1))),
    }


def model_info(yaml_path: str):
    m = YOLO(yaml_path)
    _, params, _, flops = m.model.info()
    return flops, params


def write_summary(trials_csv: str, summary_csv: str):
    with open(trials_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    rows = sorted(rows, key=lambda r: float(r["mAP50_95"]), reverse=True)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "sci_name",
                "seed",
                "epochs",
                "mAP50",
                "mAP50_95",
                "best_mAP50",
                "best_mAP50_95",
                "Params_M",
                "FLOPs_G",
                "RunTime_s",
                "save_dir",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "variant": r["variant"],
                    "sci_name": r["sci_name"],
                    "seed": r["seed"],
                    "epochs": r["epochs"],
                    "mAP50": r["mAP50"],
                    "mAP50_95": r["mAP50_95"],
                    "best_mAP50": r["best_mAP50"],
                    "best_mAP50_95": r["best_mAP50_95"],
                    "Params_M": r["Params_M"],
                    "FLOPs_G": r["FLOPs_G"],
                    "RunTime_s": r["RunTime_s"],
                    "save_dir": r["save_dir"],
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(description="SCI1 strict comparison for seed=3")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=("0" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--data", type=str, default=os.path.join(BACKUP_ROOT, "cotton.yaml"))
    parser.add_argument("--project", type=str, default=os.path.join(THIS_DIR, "outputs", "seed3_compare"))
    parser.add_argument("--force", action="store_true", help="Rerun even if entries already exist in trials.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_dir = os.path.join(THIS_DIR, "configs")
    summary_dir = os.path.join(args.project, "summary")
    trials_csv = os.path.join(summary_dir, "trials.csv")
    summary_csv = os.path.join(summary_dir, "summary.csv")

    variants: List[Variant] = [
        Variant(
            name="baseline",
            sci_name="YOLO11n-Baseline",
            yaml_path=os.path.join(cfg_dir, "yolo11n_baseline.yaml"),
            use_wiou=False,
        ),
        Variant(
            name="drbn_wiou",
            sci_name="DRBNWIoU",
            yaml_path=os.path.join(cfg_dir, "DRBN_WIoU.yaml"),
            use_wiou=True,
        ),
        Variant(
            name="drbn_wiou_tlpn",
            sci_name="DRBNWIoU-TLPN",
            yaml_path=os.path.join(cfg_dir, "DRBNWIoU_TLPN.yaml"),
            use_wiou=True,
        ),
    ]

    os.makedirs(args.project, exist_ok=True)
    ensure_csv(trials_csv)
    done = load_done(trials_csv)

    tasks.parse_model = sdp_utils.parse_model
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    for v in variants:
        key = (v.name, str(args.seed))
        if (not args.force) and key in done:
            print("skip", key)
            continue

        set_wiou(v.use_wiou)
        flops, params = model_info(v.yaml_path)

        run_name = f"SCI1_{v.name}_seed{args.seed}_{tag}"
        model = YOLO(v.yaml_path)

        t0 = time.perf_counter()
        res = model.train(
            data=args.data,
            epochs=args.epochs,
            patience=0,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer="SGD",
            close_mosaic=0,
            workers=args.workers,
            lrf=0.01,
            weight_decay=0.0005,
            momentum=0.937,
            warmup_momentum=0.8,
            lr0=0.01,
            seed=args.seed,
            deterministic=True,
            project=args.project,
            name=run_name,
            device=args.device,
            pretrained=True,
            amp=True,
        )
        t1 = time.perf_counter()

        save_dir = str(res.save_dir if res is not None else model.trainer.save_dir)
        results_csv = os.path.join(save_dir, "results.csv")
        weights = os.path.join(save_dir, "weights", "best.pt")

        last = read_metrics(results_csv)
        best = best_metrics(results_csv)

        row = {
            "variant": v.name,
            "sci_name": v.sci_name,
            "seed": args.seed,
            "epochs": args.epochs,
            "use_wiou": int(v.use_wiou),
            "yaml_path": v.yaml_path,
            "P": f"{last['P']:.5f}",
            "R": f"{last['R']:.5f}",
            "mAP50": f"{last['mAP50']:.5f}",
            "mAP50_95": f"{last['mAP50_95']:.5f}",
            "best_mAP50": f"{best['best_mAP50']:.5f}",
            "best_mAP50_95": f"{best['best_mAP50_95']:.5f}",
            "best_epoch": best["best_epoch"],
            "Params_M": f"{params/1e6:.3f}",
            "FLOPs_G": f"{flops:.3f}",
            "RunTime_s": f"{(t1 - t0):.2f}",
            "run_name": run_name,
            "save_dir": save_dir,
            "results_csv": results_csv,
            "weights": weights,
        }
        append_row(trials_csv, row)
        print("done", v.name)

    write_summary(trials_csv, summary_csv)
    print("saved", trials_csv)
    print("saved", summary_csv)


if __name__ == "__main__":
    main()
