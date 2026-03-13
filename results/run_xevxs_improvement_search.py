import csv
import os
import sys
import time
from datetime import datetime

from ultralytics import YOLO
import ultralytics.nn.tasks as tasks

ROOT = "/home/tb206/yolo_agri1/github_paper_package"
SCI1_DIR = os.path.join(ROOT, "SCI1")
CFG_DIR = os.path.join(SCI1_DIR, "configs")
BACKUP_ROOT = os.path.join(ROOT, "backupyolo")

if BACKUP_ROOT not in sys.path:
    sys.path.insert(0, BACKUP_ROOT)

import sdp_loss  # noqa: E402
import sdp_utils  # noqa: E402


def set_wiou(enabled: bool):
    if enabled:
        sdp_loss.patch_loss()
    else:
        if hasattr(sdp_loss, "restore_loss"):
            sdp_loss.restore_loss()


def read_best(results_csv: str):
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0.0, -1

    def v(r):
        return float(r.get("metrics/mAP50-95(B)") or r.get("metrics/mAP50-95") or 0.0)

    best = max(rows, key=v)
    return v(best), int(float(best.get("epoch", -1)))


def main():
    data_yaml = os.path.join(CFG_DIR, "data_cotton_xevxs_v1_local.yaml")
    out_root = "/home/tb206/yolo_agri1/github_paper_package/SCI1/outputs/xevxs_search_stage1"
    os.makedirs(out_root, exist_ok=True)

    summary_csv = "/home/tb206/yolo_agri1/github_paper_package/results/xevxs_search_stage1.csv"
    if not os.path.exists(summary_csv):
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "trial",
                    "yaml",
                    "use_wiou",
                    "seed",
                    "epochs",
                    "imgsz",
                    "lr0",
                    "lrf",
                    "weight_decay",
                    "momentum",
                    "close_mosaic",
                    "cos_lr",
                    "best_mAP50_95",
                    "best_epoch",
                    "runtime_s",
                    "save_dir",
                ],
            )
            w.writeheader()

    tasks.parse_model = sdp_utils.parse_model

    # quick screening: fewer epochs on seed=3
    seed = 3
    epochs = 120

    candidates = [
        {
            "trial": "A_drbn_wiou_default",
            "yaml": os.path.join(CFG_DIR, "DRBN_WIoU.yaml"),
            "use_wiou": True,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "close_mosaic": 0,
            "cos_lr": False,
        },
        {
            "trial": "B_tlpn_default",
            "yaml": os.path.join(CFG_DIR, "DRBNWIoU_TLPN.yaml"),
            "use_wiou": True,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "close_mosaic": 0,
            "cos_lr": False,
        },
        {
            "trial": "C_drbn_wiou_coslr",
            "yaml": os.path.join(CFG_DIR, "DRBN_WIoU.yaml"),
            "use_wiou": True,
            "imgsz": 640,
            "lr0": 0.008,
            "lrf": 0.05,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "close_mosaic": 0,
            "cos_lr": True,
        },
        {
            "trial": "D_drbn_wiou_closemosaic10",
            "yaml": os.path.join(CFG_DIR, "DRBN_WIoU.yaml"),
            "use_wiou": True,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "close_mosaic": 10,
            "cos_lr": False,
        },
        {
            "trial": "E_drbn_wiou_img704",
            "yaml": os.path.join(CFG_DIR, "DRBN_WIoU.yaml"),
            "use_wiou": True,
            "imgsz": 704,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "momentum": 0.937,
            "close_mosaic": 0,
            "cos_lr": False,
        },
        {
            "trial": "F_drbn_wiou_lowwd",
            "yaml": os.path.join(CFG_DIR, "DRBN_WIoU.yaml"),
            "use_wiou": True,
            "imgsz": 640,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0003,
            "momentum": 0.937,
            "close_mosaic": 0,
            "cos_lr": False,
        },
    ]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for c in candidates:
        set_wiou(c["use_wiou"])
        model = YOLO(c["yaml"])
        run_name = f"{c['trial']}_seed{seed}_e{epochs}_{stamp}"

        t0 = time.perf_counter()
        res = model.train(
            data=data_yaml,
            epochs=epochs,
            patience=0,
            imgsz=c["imgsz"],
            batch=16,
            optimizer="SGD",
            close_mosaic=c["close_mosaic"],
            workers=4,
            lrf=c["lrf"],
            weight_decay=c["weight_decay"],
            momentum=c["momentum"],
            warmup_momentum=0.8,
            lr0=c["lr0"],
            cos_lr=c["cos_lr"],
            seed=seed,
            deterministic=True,
            project=out_root,
            name=run_name,
            device="0",
            pretrained=True,
            amp=True,
        )
        t1 = time.perf_counter()

        save_dir = str(res.save_dir)
        best_map, best_epoch = read_best(os.path.join(save_dir, "results.csv"))

        row = {
            "trial": c["trial"],
            "yaml": c["yaml"],
            "use_wiou": int(c["use_wiou"]),
            "seed": seed,
            "epochs": epochs,
            "imgsz": c["imgsz"],
            "lr0": c["lr0"],
            "lrf": c["lrf"],
            "weight_decay": c["weight_decay"],
            "momentum": c["momentum"],
            "close_mosaic": c["close_mosaic"],
            "cos_lr": int(c["cos_lr"]),
            "best_mAP50_95": f"{best_map:.5f}",
            "best_epoch": best_epoch,
            "runtime_s": f"{(t1 - t0):.2f}",
            "save_dir": save_dir,
        }

        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            w.writerow(row)

        print("done", c["trial"], "best_mAP50_95=", f"{best_map:.5f}")

    print("saved:", summary_csv)


if __name__ == "__main__":
    main()
