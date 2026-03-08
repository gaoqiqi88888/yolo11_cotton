# GitHub Release Package (SCI Cotton Detection Study)

This package contains the minimal code/configs and key CSV results needed to reproduce the current paper experiments.

## Structure

- `SCI1/`
  - `run_sci1_seed3_compare.py` (one-click 3-variant benchmark)
  - `requirements.txt`
  - `configs/`
    - model YAMLs: `yolo11n_baseline.yaml`, `DRBN_WIoU.yaml`, `DRBNWIoU_TLPN.yaml`
    - dataset templates: `data_cotton_real_template.yaml`, `data_cotton_xevxs_v1_template.yaml`, `data_plantdoc_template.yaml`
- `backupyolo/`
  - `sdp_loss.py`, `sdp_utils.py`, `sdp_modules.py` (required custom modules)
- `results/`
  - per-dataset `summary/trials` exports
  - deduplicated `cotton_xevxs_v1` result files
  - `main_results_seed3_last.csv` (combined seed-3 table)

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r SCI1/requirements.txt
```

## Run Commands

From repo root:

```bash
python -u SCI1/run_sci1_seed3_compare.py --seed 3 --epochs 300 --data SCI1/configs/data_cotton_real_template.yaml --project SCI1/outputs/seed3_compare
python -u SCI1/run_sci1_seed3_compare.py --seed 3 --epochs 300 --data SCI1/configs/data_cotton_xevxs_v1_template.yaml --project SCI1/outputs/cotton_xevxs_v1_seed3_compare_e300
python -u SCI1/run_sci1_seed3_compare.py --seed 3 --epochs 300 --data SCI1/configs/data_plantdoc_template.yaml --project SCI1/outputs/plantdoc_seed3_compare_e300
```

## Notes

1. Edit dataset template `path` fields to your local dataset locations before running.
2. `run_sci1_seed3_compare.py` expects `backupyolo/` at repo root.
3. Large artifacts (`weights/`, full run folders) are intentionally not included here; only summary CSVs are included for manuscript traceability.

## Citation / Data Source

The external Roboflow dataset used in this study:
- Cotton disease detection-xevxs v1 (CC BY 4.0)
- https://universe.roboflow.com/disease-detection-wounm/cotton-disease-detection-xevxs/dataset/1
