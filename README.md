# Oil Well Fault Detection System

An ML early-warning system that detects operational faults in offshore oil wells from real
sensor time-series ([Petrobras 3W dataset](https://github.com/petrobras/3W)). Models
(XGBoost and a PyTorch sequence model) are served through a FastAPI service and a live
Streamlit dashboard raises alerts while a fault is still developing.

> Status: work in progress — project scaffold (Task 1 of 22).

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

## Project layout

```
src/owfd/      config, data, features, models, inference, api, dashboard, simulator
configs/       YAML configuration
notebooks/     exploration notebooks
tests/         pytest suite
docker/        Dockerfiles
data/, artifacts/   local only, gitignored
```
