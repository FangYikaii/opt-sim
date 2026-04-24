from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

import torch
from torch import nn


def _spectral_linear(in_features: int, out_features: int) -> nn.Linear:
    return nn.utils.spectral_norm(nn.Linear(in_features, out_features))


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _spectral_linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class LabRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = [LinearBlock(3, 256)]
        blocks.extend(LinearBlock(256, 256) for _ in range(6))
        self.features = nn.Sequential(*blocks)
        self.output = _spectral_linear(256, 3)

    def forward(self, design_norm: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(design_norm))


class Generator(nn.Module):
    def __init__(self, noise_dim: int = 2) -> None:
        super().__init__()
        self.noise_up = LinearBlock(noise_dim, 128)
        self.lab_up = LinearBlock(3, 128)
        regressor_layers: list[nn.Module] = []
        regressor_layers.extend(LinearBlock(256 if index == 0 else 256, 256) for index in range(8))
        regressor_layers.append(LinearBlock(256, 128))
        self.regressor = nn.Sequential(*regressor_layers)
        self.output = nn.Sequential(
            _spectral_linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, lab_norm: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        noise_features = self.noise_up(noise)
        lab_features = self.lab_up(lab_norm)
        hidden = self.regressor(torch.cat([noise_features, lab_features], dim=1))
        return self.output(hidden)


class Evaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = [LinearBlock(3, 256)]
        blocks.extend(LinearBlock(256, 256) for _ in range(3))
        blocks.append(LinearBlock(256, 128))
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(p=0.9)
        self.output = _spectral_linear(128, 1)

    def forward(self, design_norm: torch.Tensor) -> torch.Tensor:
        hidden = self.features(design_norm)
        return self.output(self.dropout(hidden))


Discriminator = Evaluator


@dataclass
class CganBundle:
    generator: Generator
    lab_regressor: LabRegressor
    lab_min: np.ndarray
    lab_max: np.ndarray
    design_min: np.ndarray
    design_max: np.ndarray
    device: str
    noise_dim: int
    seed: int
    losses: list[dict[str, float]] = field(default_factory=list)
    best_generator_state_dict: dict[str, torch.Tensor] | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_metric_name: str | None = None
    selected_checkpoint_metric_value: float | None = None


def _normalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return (values - min_values) / np.maximum(max_values - min_values, 1e-8)


def _denormalize(values: np.ndarray, min_values: np.ndarray, max_values: np.ndarray) -> np.ndarray:
    return values * (max_values - min_values) + min_values


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


def _train_lab_regressor(
    *,
    design_norm: torch.Tensor,
    lab_norm: torch.Tensor,
    epochs: int,
    batch_size: int,
    runtime_device: torch.device,
) -> LabRegressor:
    regressor = LabRegressor().to(runtime_device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()
    effective_batch_size = min(batch_size, len(design_norm))

    regressor.train()
    for _ in range(epochs):
        for batch_indices in _iter_batch_indices(len(design_norm), effective_batch_size, runtime_device):
            predicted_lab = regressor(design_norm[batch_indices])
            loss = loss_fn(predicted_lab, lab_norm[batch_indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    regressor.eval()
    for parameter in regressor.parameters():
        parameter.requires_grad_(False)
    return regressor


def fit_lightweight_cgan(
    lab_samples: np.ndarray,
    design_samples: np.ndarray,
    *,
    epochs: int = 80,
    batch_size: int = 64,
    noise_dim: int = 2,
    learning_rate: float = 0.002,
    seed: int = 7,
    record_losses: bool = False,
    device: str | None = None,
    regressor_epochs: int | None = None,
    checkpoint_metric_fn: Callable[[CganBundle, int], float | None] | None = None,
    checkpoint_metric_name: str | None = None,
    checkpoint_metric_mode: str = "min",
    checkpoint_metric_interval: int | None = None,
) -> CganBundle:
    runtime_device = get_torch_device(device)
    torch.manual_seed(seed)
    if runtime_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    lab_min = lab_samples.min(axis=0)
    lab_max = lab_samples.max(axis=0)
    design_min = design_samples.min(axis=0)
    design_max = design_samples.max(axis=0)

    lab_norm = torch.tensor(
        _normalize(lab_samples, lab_min, lab_max),
        dtype=torch.float32,
        device=runtime_device,
    )
    design_norm = torch.tensor(
        _normalize(design_samples, design_min, design_max),
        dtype=torch.float32,
        device=runtime_device,
    )

    effective_batch_size = min(batch_size, len(lab_samples))
    lab_regressor = _train_lab_regressor(
        design_norm=design_norm,
        lab_norm=lab_norm,
        epochs=regressor_epochs or max(10, min(epochs, 100)),
        batch_size=effective_batch_size,
        runtime_device=runtime_device,
    )

    generator = Generator(noise_dim=noise_dim).to(runtime_device)
    evaluator = Evaluator().to(runtime_device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    e_optimizer = torch.optim.Adam(evaluator.parameters(), lr=learning_rate / 5.0, betas=(0.5, 0.999))
    mse_loss = nn.MSELoss()

    losses: list[dict[str, float]] = []
    best_generator_state_dict: dict[str, torch.Tensor] | None = None
    selected_checkpoint_epoch: int | None = None
    selected_checkpoint_metric_value: float | None = None
    generator.train()
    evaluator.train()

    for epoch in range(1, epochs + 1):
        alpha = min(1.0, epoch / 20000.0)
        epoch_evaluator_loss = 0.0
        epoch_generator_loss = 0.0
        epoch_real_score = 0.0
        epoch_fake_score = 0.0
        epoch_lab_mse = 0.0
        batch_counter = 0

        for batch_indices in _iter_batch_indices(len(lab_norm), effective_batch_size, runtime_device):
            lab_batch = lab_norm[batch_indices]
            real_design = design_norm[batch_indices]
            batch_size_now = len(batch_indices)

            noise_eval = torch.randn(batch_size_now, noise_dim, device=runtime_device)
            fake_design_eval = generator(lab_batch, noise_eval).detach()
            real_scores = evaluator(real_design)
            fake_scores_eval = evaluator(fake_design_eval)
            evaluator_real_loss = torch.relu(1.0 - real_scores).mean()
            evaluator_fake_loss = torch.relu(1.0 + fake_scores_eval).mean()
            evaluator_loss = evaluator_real_loss + evaluator_fake_loss
            e_optimizer.zero_grad()
            evaluator_loss.backward()
            e_optimizer.step()

            noise_gen = torch.randn(batch_size_now, noise_dim, device=runtime_device)
            fake_design_gen = generator(lab_batch, noise_gen)
            fake_scores_gen = evaluator(fake_design_gen)
            predicted_lab = lab_regressor(fake_design_gen)
            lab_mse = mse_loss(predicted_lab, lab_batch)
            generator_loss = -fake_scores_gen.mean() + alpha * lab_mse
            g_optimizer.zero_grad()
            generator_loss.backward()
            g_optimizer.step()

            epoch_evaluator_loss += float(evaluator_loss.detach().cpu().item())
            epoch_generator_loss += float(generator_loss.detach().cpu().item())
            epoch_real_score += float(real_scores.detach().mean().cpu().item())
            epoch_fake_score += float(fake_scores_gen.detach().mean().cpu().item())
            epoch_lab_mse += float(lab_mse.detach().cpu().item())
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
            generator.eval()
            evaluator.eval()
            metric_bundle = CganBundle(
                generator=generator,
                lab_regressor=lab_regressor,
                lab_min=lab_min,
                lab_max=lab_max,
                design_min=design_min,
                design_max=design_max,
                device=str(runtime_device),
                noise_dim=noise_dim,
                seed=seed,
                losses=losses,
            )
            metric_value = checkpoint_metric_fn(metric_bundle, epoch)
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
            generator.train()
            evaluator.train()

    if best_generator_state_dict is None:
        best_generator_state_dict = _snapshot_state_dict(generator)
        if checkpoint_metric_fn is None:
            selected_checkpoint_epoch = epochs

    return CganBundle(
        generator=generator,
        lab_regressor=lab_regressor,
        lab_min=lab_min,
        lab_max=lab_max,
        design_min=design_min,
        design_max=design_max,
        device=str(runtime_device),
        noise_dim=noise_dim,
        seed=seed,
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

    generator = Generator(noise_dim=noise_dim).to(runtime_device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    lab_regressor = LabRegressor().to(runtime_device)
    regressor_state = checkpoint.get("lab_regressor_state_dict")
    if regressor_state is not None:
        lab_regressor.load_state_dict(regressor_state)
    lab_regressor.eval()
    for parameter in lab_regressor.parameters():
        parameter.requires_grad_(False)

    return CganBundle(
        generator=generator,
        lab_regressor=lab_regressor,
        lab_min=np.array(checkpoint["lab_min"], dtype=np.float64),
        lab_max=np.array(checkpoint["lab_max"], dtype=np.float64),
        design_min=np.array(checkpoint["design_min"], dtype=np.float64),
        design_max=np.array(checkpoint["design_max"], dtype=np.float64),
        device=str(runtime_device),
        noise_dim=noise_dim,
        seed=int(checkpoint.get("seed", 0)),
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
) -> np.ndarray:
    runtime_device = get_torch_device(device or bundle.device)
    active_noise_dim = noise_dim or bundle.noise_dim
    if seed is not None:
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
    bundle.generator = bundle.generator.to(runtime_device)
    bundle.generator.eval()
    target_norm = _normalize(target_lab.reshape(1, -1), bundle.lab_min, bundle.lab_max)
    lab_tensor = torch.tensor(
        np.repeat(target_norm, sample_count, axis=0),
        dtype=torch.float32,
        device=runtime_device,
    )
    noise = torch.randn(sample_count, active_noise_dim, device=runtime_device)
    with torch.no_grad():
        generated_norm = bundle.generator(lab_tensor, noise).detach().cpu().numpy()
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
) -> np.ndarray:
    runtime_device = get_torch_device(device or bundle.device)
    active_noise_dim = noise_dim or bundle.noise_dim
    if seed is not None:
        torch.manual_seed(seed)
        if runtime_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    bundle.generator = bundle.generator.to(runtime_device)
    bundle.generator.eval()

    target_norm = _normalize(target_labs, bundle.lab_min, bundle.lab_max)
    lab_tensor = torch.tensor(target_norm, dtype=torch.float32, device=runtime_device)
    repeated_lab_tensor = torch.repeat_interleave(lab_tensor, repeats=sample_count, dim=0)
    noise = torch.randn(len(repeated_lab_tensor), active_noise_dim, device=runtime_device)

    with torch.no_grad():
        generated_norm = bundle.generator(repeated_lab_tensor, noise).detach().cpu().numpy()

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
