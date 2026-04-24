# Backend Prototype

Minimal FastAPI backend that mirrors the current frontend workspace contracts.

## Install Python Dependencies

```bash
cd /home/fangyikai/code/opt-sim
conda env create -f environment.yml
conda activate opt_sim
```

If you already created the environment before, just activate it:

```bash
conda activate opt_sim
```

## Run

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
uvicorn backend.app.main:app --reload --port 8000
```

The backend is now GPU-aware for the lightweight cGAN path. By default it uses CUDA when `torch.cuda.is_available()` is true and falls back to CPU otherwise.

To verify the runtime:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 - <<'PY'
from backend.app.algorithms.cgan import get_torch_runtime_info
print(get_torch_runtime_info())
PY
```

Optional overrides:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
OPT_SIM_TORCH_DEVICE=auto python3 -m uvicorn backend.app.main:app --reload --port 8000
OPT_SIM_TORCH_DEVICE=cuda python3 backend/scripts/train_cgan_reproduction.py --device cuda
```

## Paper-Reproduction Workflow

The current priority is the reference-paper reproduction workflow. The repository now supports two modes:

- `synthetic`: local smoke tests on generated Ag-SiO2-Ag samples
- `paper`: direct training and evaluation on the paper's released `training set.csv` and `testing set.csv`

Synthetic smoke run:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 backend/scripts/train_cgan_reproduction.py
```

Paper-data reproduction run:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --paper-train-csv /path/to/training\ set.csv \
  --paper-test-csv /path/to/testing\ set.csv
```

Default outputs are written to `backend/artifacts/cgan_reproduction/`:

- `loss_curve.png`: generator/discriminator loss curve.
- `sampling_comparison.png`: target color vs. nearest retrieval vs. best cGAN sample.
- `candidate_diversity.png`: generated candidate spread in thickness-parameter space.
- `loss_history.csv`: per-epoch losses.
- `candidate_samples.csv`: ranked retrieval and cGAN candidates.
- `metrics.json`: dataset, training, DeltaE, diversity summary, and paper-style reproduction metrics such as JSD / solution groups when `--dataset-source paper` is used.
- `generator_checkpoint.pt`: trained generator checkpoint and normalization metadata.

## Export OpenAPI

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 backend/scripts/export_openapi.py
```

## Current API

- `GET /api/health`
- `GET /api/runs`
- `GET /api/workspace`
- `GET /api/runs/{run_id}/workspace`
- `GET /api/artifacts/{artifact_id}`
