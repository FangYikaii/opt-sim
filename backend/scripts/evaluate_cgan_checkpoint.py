from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.train_cgan_reproduction import (
    _build_progress_logger,
    _default_progress_interval,
    compact_reproduction_metrics_for_json,
    evaluate_saved_checkpoint,
    load_runtime_dependencies,
    write_paper_reproduction_details,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved cGAN checkpoint on a paper-reproduction test split.",
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--paper-test-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-lab", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    load_runtime_dependencies()

    progress_logger = _build_progress_logger()
    progress_interval = _default_progress_interval(
        total_steps=sum(1 for _ in args.paper_test_csv.open("r", encoding="utf-8-sig")) - 1,
        target_updates=20,
        max_interval=500,
    )
    bundle, reproduction_metrics = evaluate_saved_checkpoint(
        checkpoint_path=args.checkpoint_path,
        paper_test_csv=args.paper_test_csv,
        samples_per_lab=args.samples_per_lab,
        seed=args.seed,
        device=args.device,
        progress_callback=progress_logger,
        progress_interval=progress_interval,
        progress_phase="eval_only",
    )

    payload = {
        "checkpoint_path": str(args.checkpoint_path),
        "paper_test_csv": str(args.paper_test_csv),
        "samples_per_lab": args.samples_per_lab,
        "seed": args.seed,
        "device": bundle.device,
        "training": {
            "noise_dim": bundle.noise_dim,
            "generator_learning_rate": bundle.generator_learning_rate,
            "discriminator_learning_rate": bundle.discriminator_learning_rate,
            "steps_per_batch": bundle.steps_per_batch,
            "generator_hidden_dim": bundle.generator_hidden_dim,
            "generator_depth": bundle.generator_depth,
            "discriminator_hidden_dim": bundle.discriminator_hidden_dim,
            "discriminator_depth": bundle.discriminator_depth,
            "discriminator_conditioning": bundle.discriminator_conditioning,
            "alpha_start": bundle.alpha_start,
            "alpha_ramp_epochs": bundle.alpha_ramp_epochs,
            "max_alpha": bundle.max_alpha,
            "lab_delta_e_weight": bundle.lab_delta_e_weight,
            "mode_seeking_weight": bundle.mode_seeking_weight,
            "best_checkpoint_epoch": bundle.selected_checkpoint_epoch,
            "best_checkpoint_metric_name": bundle.selected_checkpoint_metric_name,
            "best_checkpoint_metric_value": bundle.selected_checkpoint_metric_value,
        },
        "paper_reproduction": compact_reproduction_metrics_for_json(reproduction_metrics),
    }

    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_paper_reproduction_details(
        reproduction_metrics,
        args.output_dir / "paper_reproduction_details.npz",
    )


if __name__ == "__main__":
    main()
