from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


TRAIN_PROGRESS_RE = re.compile(
    r"Train (?P<epoch>\d+)/\d+ \| "
    r"d_loss=(?P<d_loss>[-+]?\d*\.?\d+) \| "
    r"g_loss=(?P<g_loss>[-+]?\d*\.?\d+) \| "
    r"lab_mse=(?P<lab_mse>[-+]?\d*\.?\d+) \| "
    r"alpha=(?P<alpha>[-+]?\d*\.?\d+)"
)
CHECKPOINT_LINE_RE = re.compile(
    r"Checkpoint evaluation complete at epoch (?P<epoch>\d+) \| "
    r"paper_reproduction\.checkpoint_score=(?P<score>[-+]?\d*\.?\d+)"
)
BEST_LINE_RE = re.compile(
    r"Persisted best checkpoint "
    r"\(epoch=(?P<epoch>\d+), "
    r"paper_reproduction\.checkpoint_score=(?P<score>[-+]?\d*\.?\d+), "
    r"mean_best_delta_e=(?P<mean_best_delta_e>[-+]?\d*\.?\d+), "
    r"median_best_delta_e=(?P<median_best_delta_e>[-+]?\d*\.?\d+), "
    r"d2_within_5nm=(?P<d2_within_5nm>[-+]?\d*\.?\d+)\)"
)
TRAINING_COMPLETE_RE = re.compile(
    r"Training complete in (?P<duration>.+) \| "
    r"best_checkpoint_epoch=(?P<best_epoch>\d+) \| "
    r"paper_reproduction\.checkpoint_score=(?P<best_score>[-+]?\d*\.?\d+)"
)


def import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_checkpoint_log(log_path: Path) -> dict[str, object]:
    checkpoints: list[dict[str, float | int | bool]] = []
    persisted_by_epoch: dict[int, dict[str, float]] = {}
    training_complete: dict[str, object] | None = None

    for line in log_path.read_text(encoding="utf-8").splitlines():
        best_match = BEST_LINE_RE.search(line)
        if best_match is not None:
            epoch = int(best_match.group("epoch"))
            persisted_by_epoch[epoch] = {
                "score": float(best_match.group("score")),
                "mean_best_delta_e": float(best_match.group("mean_best_delta_e")),
                "median_best_delta_e": float(best_match.group("median_best_delta_e")),
                "d2_within_5nm": float(best_match.group("d2_within_5nm")),
            }
            continue

        checkpoint_match = CHECKPOINT_LINE_RE.search(line)
        if checkpoint_match is not None:
            epoch = int(checkpoint_match.group("epoch"))
            persisted = persisted_by_epoch.get(epoch)
            row: dict[str, float | int | bool] = {
                "epoch": epoch,
                "score": float(checkpoint_match.group("score")),
                "is_best": persisted is not None,
            }
            if persisted is not None:
                row.update(persisted)
            checkpoints.append(row)
            continue

        training_complete_match = TRAINING_COMPLETE_RE.search(line)
        if training_complete_match is not None:
            training_complete = {
                "duration": training_complete_match.group("duration"),
                "best_epoch": int(training_complete_match.group("best_epoch")),
                "best_score": float(training_complete_match.group("best_score")),
            }

    return {
        "checkpoints": checkpoints,
        "training_complete": training_complete,
    }


def parse_train_progress(log_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = TRAIN_PROGRESS_RE.search(line)
        if match is None:
            continue
        rows.append({key: float(value) for key, value in match.groupdict().items()})
    return rows


def load_loss_history(loss_history_path: Path) -> list[dict[str, float]]:
    with loss_history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                key: float(value)
                for key, value in row.items()
                if value is not None and value != ""
            }
            for row in reader
        ]


def write_analysis_json(
    *,
    parsed_log: dict[str, object],
    loss_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    payload = {
        "training_complete": parsed_log.get("training_complete"),
        "checkpoint_metrics": parsed_log.get("checkpoints"),
        "loss_history_summary": {
            "epochs": len(loss_history),
            "min_lab_mse": min((row.get("lab_mse", float("inf")) for row in loss_history), default=None),
            "max_alpha": max((row.get("alpha", float("-inf")) for row in loss_history), default=None),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def plot_checkpoint_analysis(
    *,
    checkpoints: list[dict[str, float | int | bool]],
    output_path: Path,
) -> None:
    if not checkpoints:
        return

    plt = import_pyplot()
    epochs = [int(row["epoch"]) for row in checkpoints]
    scores = [float(row["score"]) for row in checkpoints]
    mean_best_delta_es = [
        float(row["mean_best_delta_e"]) if "mean_best_delta_e" in row else float("nan")
        for row in checkpoints
    ]
    d2_within_5nm = [
        float(row["d2_within_5nm"]) if "d2_within_5nm" in row else float("nan")
        for row in checkpoints
    ]
    best_epochs = [int(row["epoch"]) for row in checkpoints if bool(row.get("is_best"))]
    best_scores = [float(row["score"]) for row in checkpoints if bool(row.get("is_best"))]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    axes[0].plot(epochs, scores, marker="o", color="#1d3557", label="checkpoint score")
    if best_epochs:
        axes[0].scatter(best_epochs, best_scores, color="#d62828", s=50, label="new best")
    axes[0].set_ylabel("Checkpoint Score")
    axes[0].set_title("Checkpoint Ranking Trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, mean_best_delta_es, marker="o", color="#2a9d8f", label="mean_best_delta_e")
    axes[1].plot(epochs, d2_within_5nm, marker="s", color="#f4a261", label="d2_within_5nm")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Primary Metrics")
    axes[1].set_title("Primary Objective Metrics at Persisted Checkpoints")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_loss_alpha_analysis(
    *,
    loss_history: list[dict[str, float]],
    output_path: Path,
) -> None:
    if not loss_history:
        return

    plt = import_pyplot()
    epochs = [row["epoch"] for row in loss_history]
    alphas = [row["alpha"] for row in loss_history]
    lab_mse = [row["lab_mse"] for row in loss_history]
    generator_loss = [row.get("generator_loss", row.get("g_loss", float("nan"))) for row in loss_history]
    discriminator_loss = [row.get("discriminator_loss", row.get("d_loss", float("nan"))) for row in loss_history]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=True)

    axes[0].plot(epochs, alphas, color="#6a4c93")
    axes[0].set_ylabel("alpha")
    axes[0].set_title("Alpha Ramp")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, lab_mse, color="#1982c4", label="lab_mse")
    axes[1].plot(epochs, generator_loss, color="#ff595e", label="generator_loss")
    axes[1].set_ylabel("Generator Side")
    axes[1].set_title("Generator-Side Signals")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(epochs, discriminator_loss, color="#8ac926")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("d_loss")
    axes[2].set_title("Discriminator Loss")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a cGAN training run from logs and loss history.")
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    log_path = run_dir / "train.log"
    loss_history_path = run_dir / "loss_history.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    parsed_log = parse_checkpoint_log(log_path)
    if loss_history_path.exists():
        loss_history = load_loss_history(loss_history_path)
    else:
        loss_history = parse_train_progress(log_path)

    write_analysis_json(
        parsed_log=parsed_log,
        loss_history=loss_history,
        output_path=run_dir / "analysis_summary.json",
    )
    plot_checkpoint_analysis(
        checkpoints=list(parsed_log.get("checkpoints", [])),
        output_path=run_dir / "analysis_checkpoint_trajectory.png",
    )
    plot_loss_alpha_analysis(
        loss_history=loss_history,
        output_path=run_dir / "analysis_alpha_loss_trajectory.png",
    )


if __name__ == "__main__":
    main()
