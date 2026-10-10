# Oil Well Fault Detection System

An ML early-warning system that detects operational faults in offshore oil wells from real
sensor time-series ([Petrobras 3W dataset](https://github.com/petrobras/3W)). Models
(XGBoost and a PyTorch sequence model) are served through a FastAPI service and a live
Streamlit dashboard raises alerts while a fault is still developing.

> Status: work in progress — EDA done (Task 3 of 22).

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/nomad-gokul/Oil-Well-Fault-Detection-.git
cd Oil-Well-Fault-Detection-
make install        # core + dev tools into .venv
make install-all    # also ML, API and dashboard dependencies
make lint && make test
```

Settings live in `configs/default.yaml` and can be overridden with `OWFD_*` environment
variables (see `.env.example`).

## Data

```bash
make install-all
python -m owfd.data.download              # sparse clone of petrobras/3W into data/raw/3W (~1.8 GB)
python -m owfd.data.download --path ~/3W  # or link an existing local copy
python -m owfd.data.profile               # per-instance quality/label profile for the EDA
```

This builds `data/processed/instances.parquet`, an index of all 2,228 instances (1,119 real
from 40 wells, 1,089 simulated, 20 hand-drawn). See [docs/data_report.md](docs/data_report.md).

## Project layout

```
src/owfd/      config, data, features, models, inference, api, dashboard, simulator
configs/       YAML configuration
notebooks/     exploration notebooks
tests/         pytest suite
docker/        Dockerfiles
data/, artifacts/   local only, gitignored
```
