# Algorithm Operations Guide

This guide explains how to run the current Ag-SiO2-Ag inverse-design stack, how to judge its present quality, and how the frontend/backend integration exposes the algorithm to users.

## 1. What the algorithm does

The current algorithm combines two parts:

- A lightweight cGAN that maps target Lab color values to candidate Ag-SiO2-Ag thickness triples.
- A thin-film transfer-matrix simulation that re-evaluates each generated structure and ranks candidates by color error and process drift.

Business-wise, the system is trying to answer:

1. "Given a target structural color, what film thickness combinations should we try?"
2. "Which of those combinations are closer to the target?"
3. "Which of those combinations are less fragile if fabrication deviates slightly?"

## 2. How to interpret current quality

Use `backend/artifacts/cgan_reproduction*/metrics.json` as the source of truth.

- `paper_reproduction` records what the current run actually achieved on the testing-set-style evaluation.
- `paper_targets` records the paper reference numbers we want to approach.

If the current `paper_reproduction.mean_best_delta_e` is much larger than `paper_targets.generator_mean_best_delta_e_with_1000_z`, the pipeline is running but has not yet reached paper-level quality.

## 3. Current state of this repository

As of the artifacts already checked into the repository:

- Checkpoints already exist in multiple folders under `backend/artifacts/`.
- The saved metrics show CUDA-based runs on an NVIDIA GPU.
- The current runs are smoke-scale or partial reproduction runs, not full paper-scale training.
- The cGAN currently loses to nearest-retrieval baselines on the demo target comparisons stored in the checked-in metrics.

That means:

- The algorithm is wired up and trainable.
- The code path is GPU-aware and has been run on GPU before.
- The present checked-in artifacts should be treated as "trained and validated as a prototype", not "fully converged final model".

## 4. Start the backend

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
uvicorn backend.app.main:app --reload --port 8002
```

Health check:

```bash
curl http://127.0.0.1:8002/api/health
```

Algorithm status endpoint:

```bash
curl http://127.0.0.1:8002/api/algorithm-overview
```

## 5. Start the frontend

```bash
cd /home/fangyikai/code/opt-sim/frontend
npm run dev
```

Open:

```text
http://127.0.0.1:9002
```

The home page now shows:

- plain-language algorithm explanation
- current effect assessment
- GPU/training status
- a detailed operator guide

The workspace page now also shows:

- algorithm snapshot beside candidate review
- the same end-to-end operating steps

## 6. Submit one business request

Either use the home-page form or call the API directly:

```bash
curl -X POST http://127.0.0.1:8002/api/agent/design-run \
  -H 'Content-Type: application/json' \
  -d '{"requirementText":"Reproduce a warm copper structural color with the Ag-SiO2-Ag paper route.","targetHex":"#bf6f4f","topK":3}'
```

The backend will:

1. parse the request
2. run the inverse-design pipeline
3. produce ranked candidate structures
4. return them to the Vue workspace for inspection

## 7. Run a smoke retraining job

Use this when you want to confirm the training pipeline is still runnable:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --output-dir backend/artifacts/cgan_reproduction_smoke \
  --epochs 5 \
  --regressor-epochs 5 \
  --batch-size 512 \
  --generator-learning-rate 1e-3 \
  --discriminator-learning-rate 2e-4 \
  --steps-per-batch 1 \
  --retrieval-metric euclidean_lab \
  --paper-samples-per-lab 16 \
  --device auto
```

Expected outputs:

- `loss_history.csv`
- `candidate_samples.csv`
- `retrieval_metric_comparison.json`
- `metrics.json`
- `generator_checkpoint.pt`

## 8. Run a paper-scale attempt

Use this only when you are ready for a much longer run:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --paper-samples-per-lab 1000 \
  --epochs 100000 \
  --regressor-epochs 10000 \
  --generator-learning-rate 1e-3 \
  --discriminator-learning-rate 2e-4 \
  --steps-per-batch 1 \
  --retrieval-metric delta_e_2000 \
  --device cuda
```

After completion, compare:

- `paper_reproduction.mean_best_delta_e`
- `paper_reproduction.d2_ground_truth_within_5nm_ratio`
- `paper_reproduction.jsd`
- `retrieval_metric_comparison.json`

against the reference values in `paper_targets`.

The current artifact metadata now records:

- colorimetry data sources (`refer_data/D65.csv`, `refer_data/tristimulus.csv`)
- scaling strategy (`Lab=standardization`, `design=normalization`)
- explicit training hyperparameters
- active retrieval metric

## 8.1 Run legacy-baseline ablations one by one

When the goal is to isolate which change hurts `DeltaE` or `d2`, prefer the legacy `tune4 alpha` structure and add only one variable at a time.

Step 0, old baseline:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --experiment-preset legacy_tune4_alpha \
  --retrieval-metric delta_e_2000 \
  --output-dir backend/artifacts/cgan_reproduction_ablate_legacy_tune4_alpha
```

Step 1, only add conditional discriminator:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --experiment-preset legacy_tune4_alpha_conditional_d \
  --retrieval-metric delta_e_2000 \
  --output-dir backend/artifacts/cgan_reproduction_ablate_conditional_d
```

Step 2, then only add `noise_dim=8`:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --experiment-preset legacy_tune4_alpha_conditional_d_noise8 \
  --retrieval-metric delta_e_2000 \
  --output-dir backend/artifacts/cgan_reproduction_ablate_conditional_d_noise8
```

Step 3, finally try low-weight mode seeking:

```bash
cd /home/fangyikai/code/opt-sim
conda activate opt_sim
python3 -u backend/scripts/train_cgan_reproduction.py \
  --dataset-source paper \
  --experiment-preset legacy_tune4_alpha_conditional_d_noise8_mode_seeking_low \
  --retrieval-metric delta_e_2000 \
  --output-dir backend/artifacts/cgan_reproduction_ablate_conditional_d_noise8_mode_low
```

Notes:

- These presets keep the old `tune4 alpha` schedule (`noise_dim=2`, `alpha_ramp_epochs=40000`, `max_alpha=0.3`, no mode seeking) as the base, then add exactly one change per step.
- Explicit CLI flags still override preset values. For example, you can keep the preset but change only `--paper-samples-per-lab 256` or `--output-dir`.
- The low mode-seeking preset currently uses `--mode-seeking-weight 0.02` so that we probe diversity pressure conservatively before trying larger weights.

## 9. Where the integration now lives

Backend:

- `backend/app/algorithm_overview.py`: artifact parsing and summary generation
- `backend/app/api/routes/algorithm.py`: `/api/algorithm-overview`

Frontend:

- `frontend/src/components/AlgorithmOverviewPanel.vue`
- `frontend/src/components/OperationsGuidePanel.vue`
- `frontend/src/pages/HomePage.vue`
- `frontend/src/pages/WorkspacePage.vue`
- `frontend/src/components/InspectorPanel.vue`

## 10. Recommended next step

The current best next step is not more UI work. It is to run a longer paper-scale training job and try to close the gap between:

- current reproduction metrics
- paper target metrics

Once the cGAN starts beating retrieval more consistently, the current UI and API wiring are already in place to present those stronger results.
