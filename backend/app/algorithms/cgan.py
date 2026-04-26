from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

import numpy as np

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import torch
from torch import nn


def _spectral_linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.utils.spectral_norm(nn.Linear(in_features, out_features))


DiscriminatorConditioning = Literal["none", "target_lab"]


class GeneratorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _spectral_linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _spectral_linear(in_features, out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class LabRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = [GeneratorBlock(3, 256)]
        blocks.extend(GeneratorBlock(256, 256) for _ in range(6))
        self.features = nn.Sequential(*blocks)
        self.output = _spectral_linear(256, 3)

    def forward(self, design_norm: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(design_norm))


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 8,
        *,
        hidden_dim: int = 160,
        trunk_depth: int = 5,
        noise_hidden_dim: int | None = None,
        lab_hidden_dim: int | None = None,
        regressor_hidden_dims: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        if regressor_hidden_dims is None:
            if trunk_depth < 1:
                raise ValueError("Generator trunk_depth must be at least 1.")
            resolved_regressor_hidden_dims = [hidden_dim] * trunk_depth
        else:
            resolved_regressor_hidden_dims = [int(value) for value in regressor_hidden_dims]
            if not resolved_regressor_hidden_dims:
                raise ValueError("Generator regressor_hidden_dims must not be empty.")
            trunk_depth = len(resolved_regressor_hidden_dims)

        resolved_noise_hidden_dim = int(noise_hidden_dim or hidden_dim)
        resolved_lab_hidden_dim = int(lab_hidden_dim or hidden_dim)

        self.hidden_dim = hidden_dim
        self.trunk_depth = trunk_depth
        self.noise_hidden_dim = resolved_noise_hidden_dim
        self.lab_hidden_dim = resolved_lab_hidden_dim
        self.regressor_hidden_dims = tuple(resolved_regressor_hidden_dims)
        self.noise_up = GeneratorBlock(noise_dim, resolved_noise_hidden_dim)
        self.lab_up = GeneratorBlock(3, resolved_lab_hidden_dim)

        regressor_layers: list[nn.Module] = []
        regressor_in_features = resolved_noise_hidden_dim + resolved_lab_hidden_dim
        for regressor_out_features in resolved_regressor_hidden_dims:
            regressor_layers.append(GeneratorBlock(regressor_in_features, regressor_out_features))
            regressor_in_features = regressor_out_features
        self.regressor = nn.Sequential(*regressor_layers)
        self.output = nn.Sequential(
            _spectral_linear(regressor_in_features, 3),
            nn.Sigmoid(),
        )

    def forward(self, lab_norm: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        noise_features = self.noise_up(noise)
        lab_features = self.lab_up(lab_norm)
        hidden = self.regressor(torch.cat([noise_features, lab_features], dim=1))
        return self.output(hidden)


class Evaluator(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        trunk_depth: int = 4,
        conditioning: DiscriminatorConditioning = "target_lab",
    ) -> None:
        super().__init__()
        if trunk_depth < 1:
            raise ValueError("Discriminator trunk_depth must be at least 1.")
        if conditioning not in {"none", "target_lab"}:
            raise ValueError("Discriminator conditioning must be 'none' or 'target_lab'.")
        self.hidden_dim = hidden_dim
        self.trunk_depth = trunk_depth
        self.conditioning = conditioning
        input_dim = 3 if conditioning == "none" else 6
        blocks: list[nn.Module] = [DiscriminatorBlock(input_dim, hidden_dim)]
        blocks.extend(DiscriminatorBlock(hidden_dim, hidden_dim) for _ in range(trunk_depth - 1))
        self.features = nn.Sequential(*blocks)
        self.output = _spectral_linear(hidden_dim, 1)

    def forward(
        self,
        design_norm: torch.Tensor,
        lab_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.conditioning == "target_lab":
            if lab_norm is None:
                raise ValueError("Conditional discriminator requires target Lab inputs.")
            features = torch.cat([design_norm, lab_norm], dim=1)
        else:
            features = design_norm
        hidden = self.features(features)
        return self.output(hidden)


Discriminator = Evaluator


@dataclass
class CganBundle:
    generator: Generator
    lab_regressor: LabRegressor
    lab_mean: np.ndarray
    lab_std: np.ndarray
    design_min: np.ndarray
    design_max: np.ndarray
    device: str
    noise_dim: int
    seed: int
    generator_learning_rate: float = 1e-3
    discriminator_learning_rate: float = 2e-4
    steps_per_batch: int = 1
    generator_hidden_dim: int = 160
    generator_depth: int = 5
    discriminator_hidden_dim: int = 128
    discriminator_depth: int = 4
    discriminator_conditioning: DiscriminatorConditioning = "target_lab"
    alpha_start: float = 0.0
    alpha_ramp_epochs: int = 2000
    max_alpha: float = 1.0
    lab_delta_e_weight: float = 0.0
    mode_seeking_weight: float = 0.0
    checkpoint_format_version: int = 3
    lab_scaling_type: str = "standardization"
    design_scaling_type: str = "normalization"
    legacy_lab_min: np.ndarray | None = None
    legacy_lab_max: np.ndarray | None = None
    losses: list[dict[str, float]] = field(default_factory=list)
    best_generator_state_dict: dict[str, torch.Tensor] | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_metric_name: str | None = None
    selected_checkpoint_metric_value: float | None = None

    @property
    def lab_min(self) -> np.ndarray:
        if self.legacy_lab_min is not None:
            return self.legacy_lab_min
        return self.lab_mean - self.lab_std

    @property
    def lab_max(self) -> np.ndarray:
        if self.legacy_lab_max is not None:
            return self.legacy_lab_max
        return self.lab_mean + self.lab_std


def _normalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return (values - min_values) / np.maximum(max_values - min_values, 1e-8)


def _denormalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return values * (max_values - min_values) + min_values


def _standardize(values: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    return (values - mean_values) / np.maximum(std_values, 1e-8)


def _destandardize(values: np.ndarray, mean_values: np.ndarray, std_values: np.ndarray) -> np.ndarray:
    return values * np.maximum(std_values, 1e-8) + mean_values


def _destandardize_tensor(
    values: torch.Tensor,
    mean_values: torch.Tensor,
    std_values: torch.Tensor,
) -> torch.Tensor:
    return values * torch.clamp(std_values, min=1e-8) + mean_values


def _clip_design_bounds(
    designs: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    return np.clip(designs, lower_bounds, upper_bounds)


def _iter_batch_indices(
    sample_count: int,
    batch_size: int,
    runtime_device: torch.device,
) -> list[torch.Tensor]:
    indices = torch.randperm(sample_count, device=runtime_device)
    return [indices[start : start + batch_size] for start in range(0, sample_count, batch_size)]


def _snapshot_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _compute_alpha(
    epoch: int,
    *,
    alpha_start: float,
    max_alpha: float,
    alpha_ramp_epochs: int,
) -> float:
    if alpha_ramp_epochs <= 1:
        return float(max_alpha)
    progress = min(max(epoch, 0) / float(alpha_ramp_epochs), 1.0)
    return float(alpha_start + (max_alpha - alpha_start) * progress)


def _mode_seeking_regularization(
    design_a: torch.Tensor,
    design_b: torch.Tensor,
    noise_a: torch.Tensor,
    noise_b: torch.Tensor,
) -> torch.Tensor:
    design_distance = torch.linalg.vector_norm(design_a - design_b, dim=1)
    noise_distance = torch.linalg.vector_norm(noise_a - noise_b, dim=1)
    ratio = design_distance / torch.clamp(noise_distance, min=1e-6)
    return 1.0 / torch.clamp(ratio.mean(), min=1e-6)


def get_torch_device(preferred: str | None = None) -> torch.device:
    requested = preferred or os.getenv("OPT_SIM_TORCH_DEVICE", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "OPT_SIM_TORCH_DEVICE requests CUDA, but torch.cuda.is_available() is False."
            )
        return torch.device(requested)

    return torch.device(requested)


def get_torch_runtime_info(preferred: str | None = None) -> dict[str, object]:
    device = get_torch_device(preferred)
    info: dict[str, object] = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
        info["device_count"] = int(torch.cuda.device_count())
    return info


def _resolve_sampling_batch_size(
    total_sample_count: int,
    runtime_device: torch.device,
    max_forward_batch_size: int | None,
) -> int:
    if total_sample_count <= 0:
        return 1
    if max_forward_batch_size is not None:
        return max(1, min(int(max_forward_batch_size), total_sample_count))
    default_batch_size = 16384 if runtime_device.type == "cuda" else 65536
    return min(default_batch_size, total_sample_count)


def _train_lab_regressor(
    *,
    design_norm: torch.Tensor,
    lab_norm: torch.Tensor,
    epochs: int,
    batch_size: int,
    runtime_device: torch.device,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int | None = None,
) -> LabRegressor:
    regressor = LabRegressor().to(runtime_device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    effective_batch_size = min(batch_size, len(design_norm))
    update_interval = max(progress_interval or 0, 0)

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "regressor",
                "event": "start",
                "epoch": 0,
                "total_epochs": epochs,
                "batch_size": effective_batch_size,
                "sample_count": int(len(design_norm)),
            }
        )

    regressor.train()
    for epoch in range(1, epochs + 1):
        epoch_loss_total = 0.0
        batch_counter = 0
        for batch_indices in _iter_batch_indices(len(design_norm), effective_batch_size, runtime_device):
            predicted_lab = regressor(design_norm[batch_indices])
            loss = loss_fn(predicted_lab, lab_norm[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_total += float(loss.detach().cpu().item())
            batch_counter += 1

        if (
            progress_callback is not None
            and (
                epoch == epochs
                or (update_interval > 0 and epoch % update_interval == 0)
            )
        ):
            progress_callback(
                {
                    "stage": "regressor",
                    "event": "progress",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "loss": epoch_loss_total / max(batch_counter, 1),
                }
            )

    regressor.eval()
    for parameter in regressor.parameters():
        parameter.requires_grad_(False)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "regressor",
                "event": "complete",
                "epoch": epochs,
                "total_epochs": epochs,
            }
        )
    return regressor


def fit_lightweight_cgan(
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    *,
    epochs: int = 80,
    batch_size: int = 2048,
    noise_dim: int = 8,
    generator_learning_rate: float = 1e-3,
    discriminator_learning_rate: float = 2e-4,
    steps_per_batch: int = 1,
    generator_hidden_dim: int = 160,
    generator_depth: int = 5,
    discriminator_hidden_dim: int = 128,
    discriminator_depth: int = 4,
    discriminator_conditioning: DiscriminatorConditioning = "target_lab",
    alpha_start: float = 0.0,
    alpha_ramp_epochs: int = 2000,
    max_alpha: float = 1.0,
    lab_delta_e_weight: float = 0.0,
    mode_seeking_weight: float = 0.0,
    seed: int = 7,
    record_losses: bool = False,
    device: str | None = None,
    regressor_epochs: int | None = None,
    checkpoint_metric_fn: Callable[[CganBundle, int], float | None] | None = None,
    checkpoint_metric_name: str | None = None,
    checkpoint_metric_mode: str = "min",
    checkpoint_metric_interval: int | None = None,
    checkpoint_patience: int | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    progress_interval: int | None = None,
    regressor_progress_interval: int | None = None,
) -> CganBundle:
    if steps_per_batch < 1:
        raise ValueError("steps_per_batch must be at least 1.")
    if generator_hidden_dim < 1 or discriminator_hidden_dim < 1:
        raise ValueError("Generator and discriminator hidden dims must be at least 1.")
    if generator_depth < 1 or discriminator_depth < 1:
        raise ValueError("Generator and discriminator depths must be at least 1.")
    if discriminator_conditioning not in {"none", "target_lab"}:
        raise ValueError("discriminator_conditioning must be 'none' or 'target_lab'.")
    if alpha_start < 0:
        raise ValueError("alpha_start must be non-negative.")
    if alpha_ramp_epochs < 1:
        raise ValueError("alpha_ramp_epochs must be at least 1.")
    if max_alpha < 0:
        raise ValueError("max_alpha must be non-negative.")
    if max_alpha < alpha_start:
        raise ValueError("max_alpha must be greater than or equal to alpha_start.")
    if lab_delta_e_weight < 0:
        raise ValueError("lab_delta_e_weight must be non-negative.")
    if mode_seeking_weight < 0:
        raise ValueError("mode_seeking_weight must be non-negative.")

    runtime_device = get_torch_device(device)
    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    lab_mean = lab_samples.mean(axis=0)
    lab_std = lab_samples.std(axis=0)
    design_min = design_samples.min(axis=0)
    design_max = design_samples.max(axis=0)

    lab_norm = torch.tensor(
        _standardize(lab_samples, lab_mean, lab_std),
        dtype=torch.float32,
        device=runtime_device,
    )
    design_norm = torch.tensor(
        _normalize(design_samples, design_min, design_max),
        dtype=torch.float32,
        device=runtime_device,
    )

    effective_batch_size = min(batch_size, len(lab_samples))
    lab_mean_tensor = torch.tensor(
        lab_mean,
        dtype=torch.float32,
        device=runtime_device,
    )
    lab_std_tensor = torch.tensor(
        np.maximum(lab_std, 1e-8),
        dtype=torch.float32,
        device=runtime_device,
    )
    lab_regressor = _train_lab_regressor(
        design_norm=design_norm,
        lab_norm=lab_norm,
        epochs=regressor_epochs or max(10, min(epochs, 100)),
        batch_size=effective_batch_size,
        runtime_device=runtime_device,
        progress_callback=progress_callback,
        progress_interval=regressor_progress_interval,
    )

    generator = Generator(
        noise_dim=noise_dim,
        hidden_dim=generator_hidden_dim,
        trunk_depth=generator_depth,
    ).to(runtime_device)
    evaluator = Evaluator(
        hidden_dim=discriminator_hidden_dim,
        trunk_depth=discriminator_depth,
        conditioning=discriminator_conditioning,
    ).to(runtime_device)
    g_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=generator_learning_rate,
        betas=(0.5, 0.999),
    )
    e_optimizer = torch.optim.Adam(
        evaluator.parameters(),
        lr=discriminator_learning_rate,
        betas=(0.5, 0.999),
    )
    mse_loss = nn.MSELoss()

    losses: list[dict[str, float]] = []
    best_generator_state_dict: dict[str, torch.Tensor] | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_metric_value: float | None = None
    checkpoint_evals_without_improvement = 0
    completed_epochs = 0
    generator.train()
    evaluator.train()
    update_interval = max(progress_interval or 0, 0)

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "training",
                "event": "start",
                "epoch": 0,
                "total_epochs": epochs,
                "batch_size": effective_batch_size,
                "sample_count": int(len(lab_samples)),
                "device": str(runtime_device),
                "generator_learning_rate": float(generator_learning_rate),
                "discriminator_learning_rate": float(discriminator_learning_rate),
                "steps_per_batch": int(steps_per_batch),
                "discriminator_conditioning": discriminator_conditioning,
                "alpha_start": float(alpha_start),
                "alpha_ramp_epochs": int(alpha_ramp_epochs),
                "max_alpha": float(max_alpha),
                "lab_delta_e_weight": float(lab_delta_e_weight),
                "mode_seeking_weight": float(mode_seeking_weight),
            }
        )

    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        alpha = _compute_alpha(
            epoch,
            alpha_start=alpha_start,
            max_alpha=max_alpha,
            alpha_ramp_epochs=alpha_ramp_epochs,
        )
        epoch_evaluator_loss = 0.0
        epoch_generator_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        epoch_lab_mse = 0.0
        epoch_lab_delta_e76 = 0.0
        epoch_color_loss = 0.0
        epoch_mode_seeking_loss = 0.0
        batch_counter = 0

        for batch_indices in _iter_batch_indices(len(lab_norm), effective_batch_size, runtime_device):
            lab_batch = lab_norm[batch_indices]
            real_design = design_norm[batch_indices]
            batch_size_now = len(batch_indices)
            batch_evaluator_loss = 0.0
            batch_generator_loss = 0.0
            batch_real_score = 0.0
            batch_fake_score = 0.0
            batch_lab_mse = 0.0
            batch_lab_delta_e76 = 0.0
            batch_color_loss = 0.0
            batch_mode_seeking_loss = 0.0

            for _ in range(steps_per_batch):
                noise_eval = torch.randn(batch_size_now, noise_dim, device=runtime_device)
                fake_design_eval = generator(lab_batch, noise_eval).detach()
                real_scores = evaluator(real_design, lab_batch)
                fake_scores_eval = evaluator(fake_design_eval, lab_batch)
                evaluator_real_loss = torch.relu(1.0 - real_scores).mean()
                evaluator_fake_loss = torch.relu(1.0 + fake_scores_eval).mean()
                evaluator_loss = 0.5 * (evaluator_real_loss + evaluator_fake_loss)
                e_optimizer.zero_grad()
                evaluator_loss.backward()
                e_optimizer.step()

                noise_gen_a = torch.randn(batch_size_now, noise_dim, device=runtime_device)
                noise_gen_b = torch.randn(batch_size_now, noise_dim, device=runtime_device)
                fake_design_gen = generator(lab_batch, noise_gen_a)
                fake_design_gen_b = generator(lab_batch, noise_gen_b)
                fake_scores_gen = evaluator(fake_design_gen, lab_batch)
                predicted_lab = lab_regressor(fake_design_gen)
                lab_mse = mse_loss(predicted_lab, lab_batch)
                predicted_lab_real = _destandardize_tensor(predicted_lab, lab_mean_tensor, lab_std_tensor)
                target_lab_real = _destandardize_tensor(lab_batch, lab_mean_tensor, lab_std_tensor)
                lab_delta_e76 = torch.linalg.vector_norm(predicted_lab_real - target_lab_real, dim=1).mean()
                color_loss = lab_mse + (lab_delta_e_weight * lab_delta_e76)
                mode_seeking_loss = _mode_seeking_regularization(
                    fake_design_gen,
                    fake_design_gen_b,
                    noise_gen_a,
                    noise_gen_b,
                )
                generator_loss = (
                    -fake_scores_gen.mean()
                    + alpha * color_loss
                    + (mode_seeking_weight * mode_seeking_loss)
                )
                g_optimizer.zero_grad()
                generator_loss.backward()
                g_optimizer.step()

                batch_evaluator_loss += float(evaluator_loss.detach().cpu().item())
                batch_generator_loss += float(generator_loss.detach().cpu().item())
                batch_real_score += float(real_scores.detach().mean().cpu().item())
                batch_fake_score += float(fake_scores_gen.detach().mean().cpu().item())
                batch_lab_mse += float(lab_mse.detach().cpu().item())
                batch_lab_delta_e76 += float(lab_delta_e76.detach().cpu().item())
                batch_color_loss += float(color_loss.detach().cpu().item())
                batch_mode_seeking_loss += float(mode_seeking_loss.detach().cpu().item())

            normalizer = float(steps_per_batch)
            epoch_evaluator_loss += batch_evaluator_loss / normalizer
            epoch_generator_loss += batch_generator_loss / normalizer
            epoch_real_score += batch_real_score / normalizer
            epoch_fake_score += batch_fake_score / normalizer
            epoch_lab_mse += batch_lab_mse / normalizer
            epoch_lab_delta_e76 += batch_lab_delta_e76 / normalizer
            epoch_color_loss += batch_color_loss / normalizer
            epoch_mode_seeking_loss += batch_mode_seeking_loss / normalizer
            batch_counter += 1

        if record_losses:
            normalizer = max(batch_counter, 1)
            losses.append(
                {
                    "epoch": float(epoch),
                    "discriminator_loss": epoch_evaluator_loss / normalizer,
                    "generator_loss": epoch_generator_loss / normalizer,
                    "evaluator_loss": epoch_evaluator_loss / normalizer,
                    "real_score": epoch_real_score / normalizer,
                    "fake_score": epoch_fake_score / normalizer,
                    "lab_mse": epoch_lab_mse / normalizer,
                    "lab_delta_e76": epoch_lab_delta_e76 / normalizer,
                    "color_loss": epoch_color_loss / normalizer,
                    "mode_seeking_loss": epoch_mode_seeking_loss / normalizer,
                    "alpha": float(alpha),
                }
            )

        normalizer = max(batch_counter, 1)
        if (
            progress_callback is not None
            and (
                epoch == epochs
                or (update_interval > 0 and epoch % update_interval == 0)
            )
        ):
            progress_callback(
                {
                    "stage": "training",
                    "event": "progress",
                    "epoch": epoch,
                    "total_epochs": epochs,
                    "evaluator_loss": epoch_evaluator_loss / normalizer,
                    "generator_loss": epoch_generator_loss / normalizer,
                    "real_score": epoch_real_score / normalizer,
                    "fake_score": epoch_fake_score / normalizer,
                    "lab_mse": epoch_lab_mse / normalizer,
                    "lab_delta_e76": epoch_lab_delta_e76 / normalizer,
                    "color_loss": epoch_color_loss / normalizer,
                    "mode_seeking_loss": epoch_mode_seeking_loss / normalizer,
                    "alpha": float(alpha),
                }
            )

        should_run_checkpoint_eval = (
            checkpoint_metric_fn is not None
            and (
                epoch == epochs
                or (
                    checkpoint_metric_interval is not None
                    and checkpoint_metric_interval > 0
                    and epoch % checkpoint_metric_interval == 0
                )
            )
        )
        if should_run_checkpoint_eval:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "checkpoint",
                        "event": "start",
                        "epoch": epoch,
                        "metric_name": checkpoint_metric_name,
                    }
                )
            generator.eval()
            evaluator.eval()
            metric_bundle = CganBundle(
                generator=generator,
                lab_regressor=lab_regressor,
                lab_mean=lab_mean,
                lab_std=lab_std,
                design_min=design_min,
                design_max=design_max,
                device=str(runtime_device),
                noise_dim=noise_dim,
                seed=seed,
                generator_learning_rate=float(generator_learning_rate),
                discriminator_learning_rate=float(discriminator_learning_rate),
                steps_per_batch=int(steps_per_batch),
                generator_hidden_dim=int(generator_hidden_dim),
                generator_depth=int(generator_depth),
                discriminator_hidden_dim=int(discriminator_hidden_dim),
                discriminator_depth=int(discriminator_depth),
                discriminator_conditioning=discriminator_conditioning,
                alpha_start=float(alpha_start),
                alpha_ramp_epochs=int(alpha_ramp_epochs),
                max_alpha=float(max_alpha),
                lab_delta_e_weight=float(lab_delta_e_weight),
                mode_seeking_weight=float(mode_seeking_weight),
                losses=losses,
            )
            metric_value = checkpoint_metric_fn(metric_bundle, epoch)
            is_better = False
            if metric_value is not None and np.isfinite(metric_value):
                is_better = (
                    selected_checkpoint_metric_value is None
                    or (
                        checkpoint_metric_mode == "min"
                        and float(metric_value) < selected_checkpoint_metric_value
                    )
                    or (
                        checkpoint_metric_mode == "max"
                        and float(metric_value) > selected_checkpoint_metric_value
                    )
                )
                if is_better:
                    best_generator_state_dict = _snapshot_state_dict(generator)
                    selected_checkpoint_epoch = epoch
                    selected_checkpoint_metric_value = float(metric_value)
                    checkpoint_evals_without_improvement = 0
                else:
                    checkpoint_evals_without_improvement += 1
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "checkpoint",
                        "event": "complete",
                        "epoch": epoch,
                        "metric_name": checkpoint_metric_name,
                        "metric_value": float(metric_value) if metric_value is not None else None,
                        "is_best": is_better,
                        "checkpoint_evals_without_improvement": checkpoint_evals_without_improvement,
                    }
                )
            generator.train()
            evaluator.train()
            if (
                checkpoint_patience is not None
                and checkpoint_patience > 0
                and checkpoint_evals_without_improvement >= checkpoint_patience
            ):
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "training",
                            "event": "early_stop",
                            "epoch": epoch,
                            "total_epochs": epochs,
                            "checkpoint_patience": checkpoint_patience,
                            "checkpoint_evals_without_improvement": checkpoint_evals_without_improvement,
                            "selected_checkpoint_epoch": selected_checkpoint_epoch,
                            "selected_checkpoint_metric_name": checkpoint_metric_name,
                            "selected_checkpoint_metric_value": selected_checkpoint_metric_value,
                        }
                    )
                break

    if best_generator_state_dict is None:
        best_generator_state_dict = _snapshot_state_dict(generator)
        if checkpoint_metric_fn is None:
            selected_checkpoint_epoch = completed_epochs or epochs

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "training",
                "event": "complete",
                "epoch": completed_epochs or epochs,
                "total_epochs": epochs,
                "selected_checkpoint_epoch": selected_checkpoint_epoch,
                "selected_checkpoint_metric_name": checkpoint_metric_name,
                "selected_checkpoint_metric_value": selected_checkpoint_metric_value,
            }
        )

    return CganBundle(
        generator=generator,
        lab_regressor=lab_regressor,
        lab_mean=lab_mean,
        lab_std=lab_std,
        design_min=design_min,
        design_max=design_max,
        device=str(runtime_device),
        noise_dim=noise_dim,
        seed=seed,
        generator_learning_rate=float(generator_learning_rate),
        discriminator_learning_rate=float(discriminator_learning_rate),
        steps_per_batch=int(steps_per_batch),
        generator_hidden_dim=int(generator_hidden_dim),
        generator_depth=int(generator_depth),
        discriminator_hidden_dim=int(discriminator_hidden_dim),
        discriminator_depth=int(discriminator_depth),
        discriminator_conditioning=discriminator_conditioning,
        alpha_start=float(alpha_start),
        alpha_ramp_epochs=int(alpha_ramp_epochs),
        max_alpha=float(max_alpha),
        lab_delta_e_weight=float(lab_delta_e_weight),
        mode_seeking_weight=float(mode_seeking_weight),
        losses=losses,
        best_generator_state_dict=best_generator_state_dict,
        selected_checkpoint_epoch=selected_checkpoint_epoch,
        selected_checkpoint_metric_name=checkpoint_metric_name,
        selected_checkpoint_metric_value=selected_checkpoint_metric_value,
    )


def _load_model_bundle_from_disk(
    checkpoint_path: Path,
    *,
    device: str | None = None,
) -> CganBundle:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    runtime_device = get_torch_device(device)
    noise_dim = int(checkpoint.get("noise_dim", 2))
    checkpoint_format_version = int(checkpoint.get("checkpoint_format_version", 1))
    generator_state_dict = checkpoint["generator_state_dict"]

    inferred_generator_hidden_dim: int | None = None
    inferred_generator_depth: int | None = None
    inferred_noise_hidden_dim: int | None = None
    inferred_lab_hidden_dim: int | None = None
    inferred_regressor_hidden_dims: list[int] | None = None
    if "generator_hidden_dim" not in checkpoint:
        inferred_generator_hidden_dim = int(generator_state_dict["output.0.weight_orig"].shape[1])
    if "generator_depth" not in checkpoint:
        regressor_blocks = {
            int(key.split(".")[1])
            for key in generator_state_dict
            if key.startswith("regressor.") and key.endswith("weight_orig")
        }
        inferred_generator_depth = len(regressor_blocks)
    if "noise_hidden_dim" not in checkpoint:
        inferred_noise_hidden_dim = int(generator_state_dict["noise_up.net.0.weight_orig"].shape[0])
    if "lab_hidden_dim" not in checkpoint:
        inferred_lab_hidden_dim = int(generator_state_dict["lab_up.net.0.weight_orig"].shape[0])
    if "regressor_hidden_dims" not in checkpoint:
        regressor_block_shapes = {
            int(key.split(".")[1]): int(value.shape[0])
            for key, value in generator_state_dict.items()
            if key.startswith("regressor.") and key.endswith("weight_orig")
        }
        inferred_regressor_hidden_dims = [
            regressor_block_shapes[index]
            for index in sorted(regressor_block_shapes)
        ]

    generator_hidden_dim = int(checkpoint.get("generator_hidden_dim", inferred_generator_hidden_dim or 160))
    generator_depth = int(checkpoint.get("generator_depth", inferred_generator_depth or 5))
    discriminator_hidden_dim = int(checkpoint.get("discriminator_hidden_dim", 128))
    discriminator_depth = int(checkpoint.get("discriminator_depth", 4))

    generator = Generator(
        noise_dim=noise_dim,
        hidden_dim=generator_hidden_dim,
        trunk_depth=generator_depth,
        noise_hidden_dim=checkpoint.get("noise_hidden_dim", inferred_noise_hidden_dim),
        lab_hidden_dim=checkpoint.get("lab_hidden_dim", inferred_lab_hidden_dim),
        regressor_hidden_dims=checkpoint.get("regressor_hidden_dims", inferred_regressor_hidden_dims),
    ).to(runtime_device)
    generator.load_state_dict(generator_state_dict)
    generator.eval()

    lab_regressor = LabRegressor().to(runtime_device)
    regressor_state = checkpoint.get("lab_regressor_state_dict")
    if regressor_state is not None:
        lab_regressor.load_state_dict(regressor_state)
    lab_regressor.eval()
    for parameter in lab_regressor.parameters():
        parameter.requires_grad_(False)

    if checkpoint_format_version >= 2:
        lab_mean = np.array(checkpoint["lab_mean"], dtype=np.float64)
        lab_std = np.array(checkpoint["lab_std"], dtype=np.float64)
        lab_scaling_type = str(checkpoint.get("lab_scaling_type", "standardization"))
        design_scaling_type = str(checkpoint.get("design_scaling_type", "normalization"))
    elif "lab_min" in checkpoint and "lab_max" in checkpoint:
        lab_min = np.array(checkpoint["lab_min"], dtype=np.float64)
        lab_max = np.array(checkpoint["lab_max"], dtype=np.float64)
        lab_mean = 0.5 * (lab_min + lab_max)
        lab_std = np.maximum(lab_max - lab_min, 1e-8)
        lab_scaling_type = "min_max"
        design_scaling_type = "normalization"
    else:
        raise ValueError(
            f"Unsupported cGAN checkpoint format in {checkpoint_path}: missing Lab scaling metadata."
        )

    return CganBundle(
        generator=generator,
        lab_regressor=lab_regressor,
        lab_mean=lab_mean,
        lab_std=lab_std,
        design_min=np.array(checkpoint["design_min"], dtype=np.float64),
        design_max=np.array(checkpoint["design_max"], dtype=np.float64),
        device=str(runtime_device),
        noise_dim=noise_dim,
        seed=int(checkpoint.get("seed", 0)),
        generator_learning_rate=float(checkpoint.get("generator_learning_rate", 1e-3)),
        discriminator_learning_rate=float(checkpoint.get("discriminator_learning_rate", 2e-4)),
        steps_per_batch=int(checkpoint.get("steps_per_batch", 1)),
        generator_hidden_dim=generator_hidden_dim,
        generator_depth=generator_depth,
        discriminator_hidden_dim=discriminator_hidden_dim,
        discriminator_depth=discriminator_depth,
        discriminator_conditioning=str(
            checkpoint.get(
                "discriminator_conditioning",
                "target_lab" if checkpoint_format_version >= 3 else "none",
            )
        ),
        alpha_start=float(checkpoint.get("alpha_start", 0.0)),
        alpha_ramp_epochs=int(checkpoint.get("alpha_ramp_epochs", 20000)),
        max_alpha=float(checkpoint.get("max_alpha", 1.0)),
        lab_delta_e_weight=float(checkpoint.get("lab_delta_e_weight", 0.0)),
        mode_seeking_weight=float(checkpoint.get("mode_seeking_weight", 0.0)),
        checkpoint_format_version=checkpoint_format_version,
        lab_scaling_type=lab_scaling_type,
        design_scaling_type=design_scaling_type,
        legacy_lab_min=lab_min if checkpoint_format_version == 1 else None,
        legacy_lab_max=lab_max if checkpoint_format_version == 1 else None,
        selected_checkpoint_epoch=(
            int(checkpoint["selected_checkpoint_epoch"])
            if checkpoint.get("selected_checkpoint_epoch") is not None
            else None
        ),
        selected_checkpoint_metric_name=(
            str(checkpoint["selected_checkpoint_metric_name"])
            if checkpoint.get("selected_checkpoint_metric_name") is not None
            else None
        ),
        selected_checkpoint_metric_value=(
            float(checkpoint["selected_checkpoint_metric_value"])
            if checkpoint.get("selected_checkpoint_metric_value") is not None
            else None
        ),
    )


@lru_cache(maxsize=4)
def _load_model_bundle_cached(
    checkpoint_path: str,
    checkpoint_mtime_ns: int,
    device: str | None,
) -> CganBundle:
    _ = checkpoint_mtime_ns
    return _load_model_bundle_from_disk(Path(checkpoint_path), device=device)


def load_model_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> CganBundle:
    resolved_path = Path(checkpoint_path).resolve()
    return _load_model_bundle_cached(
        str(resolved_path),
        resolved_path.stat().st_mtime_ns,
        device,
    )


@lru_cache(maxsize=1)
def train_lightweight_cgan(
    lab_samples_key: bytes,
    design_samples_key: bytes,
    lab_shape: tuple[int, int],
    design_shape: tuple[int, int],
) -> CganBundle:
    lab_samples = np.frombuffer(lab_samples_key, dtype=np.float64).reshape(lab_shape)
    design_samples = np.frombuffer(design_samples_key, dtype=np.float64).reshape(design_shape)

    return fit_lightweight_cgan(lab_samples, design_samples)


def sample_designs_from_bundle(
    bundle: CganBundle,
    target_lab: np.ndarray,
    *,
    sample_count: int = 24,
    noise_dim: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    max_forward_batch_size: int | None = None,
) -> np.ndarray:
    runtime_device = get_torch_device(device or bundle.device)
    active_noise_dim = noise_dim or bundle.noise_dim
    if seed is not None:
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    bundle.generator = bundle.generator.to(runtime_device)
    bundle.generator.eval()
    if bundle.lab_scaling_type == "standardization":
        target_norm = _standardize(target_lab.reshape(1, -1), bundle.lab_mean, bundle.lab_std)
    else:
        target_norm = _normalize(target_lab.reshape(1, -1), bundle.lab_min, bundle.lab_max)
    if sample_count <= 0:
        return np.empty((0, len(bundle.design_min)), dtype=np.float32)
    effective_batch_size = _resolve_sampling_batch_size(
        sample_count,
        runtime_device,
        max_forward_batch_size,
    )
    noise = torch.randn(sample_count, active_noise_dim, device=runtime_device)
    generated_norm = np.empty((sample_count, len(bundle.design_min)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, sample_count, effective_batch_size):
            end = min(start + effective_batch_size, sample_count)
            lab_tensor = torch.tensor(
                np.repeat(target_norm, end - start, axis=0),
                dtype=torch.float32,
                device=runtime_device,
            )
            generated_norm[start:end] = (
                bundle.generator(lab_tensor, noise[start:end]).detach().cpu().numpy()
            )
    generated = _denormalize(generated_norm, bundle.design_min, bundle.design_max)
    return _clip_design_bounds(generated, bundle.design_min, bundle.design_max)


def sample_designs_for_labs_from_bundle(
    bundle: CganBundle,
    target_labs: np.ndarray,
    *,
    sample_count: int,
    noise_dim: int | None = None,
    seed: int | None = None,
    device: str | None = None,
    max_forward_batch_size: int | None = None,
) -> np.ndarray:
    runtime_device = get_torch_device(device or bundle.device)
    active_noise_dim = noise_dim or bundle.noise_dim
    if seed is not None:
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    bundle.generator = bundle.generator.to(runtime_device)
    bundle.generator.eval()

    if bundle.lab_scaling_type == "standardization":
        target_norm = _standardize(target_labs, bundle.lab_mean, bundle.lab_std)
    else:
        target_norm = _normalize(target_labs, bundle.lab_min, bundle.lab_max)
    if sample_count <= 0:
        return np.empty((len(target_labs), 0, len(bundle.design_min)), dtype=np.float32)
    total_sample_count = len(target_labs) * sample_count
    effective_batch_size = _resolve_sampling_batch_size(
        total_sample_count,
        runtime_device,
        max_forward_batch_size,
    )
    noise = torch.randn(total_sample_count, active_noise_dim, device=runtime_device)
    generated_norm = np.empty((total_sample_count, len(bundle.design_min)), dtype=np.float32)

    with torch.no_grad():
        for chunk_start in range(0, total_sample_count, effective_batch_size):
            chunk_end = min(chunk_start + effective_batch_size, total_sample_count)
            lab_indices = np.arange(chunk_start, chunk_end) // sample_count
            repeated_lab_tensor = torch.tensor(
                target_norm[lab_indices],
                dtype=torch.float32,
                device=runtime_device,
            )
            generated_norm[chunk_start:chunk_end] = (
                bundle.generator(repeated_lab_tensor, noise[chunk_start:chunk_end])
                .detach()
                .cpu()
                .numpy()
            )

    generated = _denormalize(generated_norm, bundle.design_min, bundle.design_max)
    clipped = _clip_design_bounds(generated, bundle.design_min, bundle.design_max)
    return clipped.reshape(len(target_labs), sample_count, -1)


def sample_designs_from_cgan(
    target_lab: np.ndarray,
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    sample_count: int = 24,
) -> np.ndarray:
    bundle = train_lightweight_cgan(
        lab_samples.tobytes(),
        design_samples.tobytes(),
        lab_samples.shape,
        design_samples.shape,
    )
    return sample_designs_from_bundle(bundle, target_lab, sample_count=sample_count)
