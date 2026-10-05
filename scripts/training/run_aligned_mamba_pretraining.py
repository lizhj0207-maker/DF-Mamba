from __future__ import annotations

# Must be set before importing torch/CUDA libraries.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import contextlib
import gc
import hashlib
import json
import math
import platform
import random
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import get_linear_schedule_with_warmup


ORIGINAL_ROOT = Path(r"D:\Capture_Mamba_Clean")
WORK_ROOT = Path(r"D:\Capture_Mamba_Paper_Study")

MODEL_CONFIG_DIR = ORIGINAL_ROOT / "pretrained_model"
FEATURE_PATH = (
    ORIGINAL_ROOT / "data" / "source_raw" / "DFFeature_large.npy"
)
SCALER_PATH = (
    ORIGINAL_ROOT / "code_pkg" / "pretrain_data_standard_minmax.sav"
)

sys.path.insert(0, str(WORK_ROOT))

from code_pkg.DF_transformer.configuration_dff import DFFConfig
from code_pkg.DF_transformer.modeling_dff_mamba_aligned import (
    DFFForImageClassification,
    DFFForPreTraining,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-data pretraining for Protocol-aligned DF-Mamba. "
            "Uses the recovered 19,513-sample PDBbind feature dictionary and "
            "never reads labels or CASF files."
        )
    )
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument(
        "--stop_after_steps",
        type=int,
        default=None,
        help=(
            "Optional controlled stopping point. The scheduler still uses "
            "--total_steps, allowing a later resume."
        ),
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=0,
        help=(
            "0 uses all 19,513 samples. Positive values select a deterministic "
            "subset for smoke/pilot tests."
        ),
    )
    parser.add_argument("--data_seed", type=int, default=2026)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "Use fp32 for the strict fair baseline. bf16 is available only "
            "for later efficiency experiments."
        ),
    )
    parser.add_argument("--scale_chunk_size", type=int, default=64)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--save_state_every",
        type=int,
        default=0,
        help=(
            "0 disables resumable optimizer-state saves. For formal training, "
            "use 500; only one latest state file is retained."
        ),
    )
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to latest_training_state.pt from the same output directory.",
    )
    args = parser.parse_args()

    if args.total_steps <= 0:
        parser.error("--total_steps must be positive")
    if args.stop_after_steps is not None:
        if args.stop_after_steps <= 0:
            parser.error("--stop_after_steps must be positive")
        if args.stop_after_steps > args.total_steps:
            parser.error("--stop_after_steps cannot exceed --total_steps")
    if args.sample_limit < 0:
        parser.error("--sample_limit cannot be negative")
    if args.micro_batch_size <= 0 or args.effective_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.effective_batch_size % args.micro_batch_size != 0:
        parser.error(
            "--effective_batch_size must be divisible by --micro_batch_size"
        )
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if not 0.0 < args.mask_ratio < 1.0:
        parser.error("--mask_ratio must be in (0, 1)")
    if min(args.d_state, args.d_conv, args.expand) <= 0:
        parser.error("Mamba structural parameters must be positive")
    if args.scale_chunk_size <= 0 or args.log_every <= 0:
        parser.error("chunk and logging intervals must be positive")
    if args.save_state_every < 0:
        parser.error("--save_state_every cannot be negative")
    if not args.output_name.strip():
        parser.error("--output_name cannot be empty")
    return args


def normalize_id(value: object) -> str:
    return str(value).strip().lower()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_weights_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def training_state_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_reproducible(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_state = state.get("torch_cuda_rng_state_all")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def require_inputs() -> None:
    for path in (
        MODEL_CONFIG_DIR / "config.json",
        FEATURE_PATH,
        SCALER_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def fixed_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "total_steps": args.total_steps,
        "sample_limit": args.sample_limit,
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_clip_norm": args.gradient_clip_norm,
        "mask_ratio": args.mask_ratio,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "precision": args.precision,
        "scale_chunk_size": args.scale_chunk_size,
        "log_every": args.log_every,
        "save_state_every": args.save_state_every,
        "output_name": args.output_name,
    }


def output_dir_for(name: str) -> Path:
    return (
        WORK_ROOT
        / "paper_outputs"
        / "03_aligned_mamba"
        / "pretraining"
        / name
    )


def prepare_new_run(args: argparse.Namespace, script_path: Path) -> Path:
    output_dir = output_dir_for(args.output_name)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}\n"
            "Use a new --output_name. Existing results will not be overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot = output_dir / "training_script_snapshot.py"
    shutil.copy2(script_path, snapshot)

    environment = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }
    manifest = {
        "status": "created",
        "scope": (
            "Aligned-Mamba self-supervised pretraining. No labels or CASF "
            "benchmark files are read."
        ),
        "script_snapshot": str(snapshot),
        "script_sha256": sha256_file(snapshot),
        "feature_path": str(FEATURE_PATH),
        "scaler_path": str(SCALER_PATH),
        "fixed_arguments": fixed_arguments(args),
        "environment": environment,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    return output_dir


def prepare_resume_run(
    args: argparse.Namespace,
    script_path: Path,
    resume_path: Path,
) -> Path:
    if not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    output_dir = resume_path.parent
    expected = output_dir_for(args.output_name).resolve()
    if output_dir.resolve() != expected:
        raise ValueError(
            f"Resume state belongs to {output_dir}, but --output_name maps "
            f"to {expected}"
        )

    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if sha256_file(script_path) != manifest["script_sha256"]:
        raise RuntimeError(
            "Current script differs from the snapshot that started this run."
        )
    if fixed_arguments(args) != manifest["fixed_arguments"]:
        raise RuntimeError(
            "Fixed arguments differ from the original invocation. Only "
            "--resume_from and --stop_after_steps may change."
        )
    return output_dir


def append_command(output_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    with (output_dir / "command_history.txt").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {command}\n"
        )


def update_manifest(output_dir: Path, **updates: Any) -> None:
    path = output_dir / "run_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    write_json(path, value)


def select_and_materialize_features(
    sample_limit: int,
    data_seed: int,
    scale_chunk_size: int,
    output_dir: Path,
) -> tuple[torch.Tensor, list[str], int]:
    print(f"Loading real pretraining feature dictionary: {FEATURE_PATH}")
    feature_dict = np.load(FEATURE_PATH, allow_pickle=True).item()
    total_available = len(feature_dict)

    key_map: dict[str, object] = {}
    for raw_key in feature_dict:
        normalized = normalize_id(raw_key)
        if normalized in key_map:
            raise ValueError(f"Duplicate normalized feature ID: {normalized}")
        key_map[normalized] = raw_key

    all_ids = np.asarray(sorted(key_map), dtype=object)
    if sample_limit == 0:
        selected_ids = all_ids.tolist()
    else:
        if sample_limit > len(all_ids):
            raise ValueError(
                f"Requested {sample_limit} samples, but only "
                f"{len(all_ids)} exist."
            )
        rng = np.random.default_rng(data_seed)
        selected_ids = rng.permutation(all_ids)[:sample_limit].tolist()

    pd.DataFrame({"normalized_id": selected_ids}).to_csv(
        output_dir / "selected_pretraining_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    n_samples = len(selected_ids)
    features = np.empty((n_samples, 6, 100, 143), dtype=np.float32)
    for index, sample_id in enumerate(selected_ids):
        array = np.asarray(feature_dict[key_map[sample_id]])
        if array.shape != (6, 100, 143):
            raise ValueError(f"{sample_id}: unexpected shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{sample_id}: NaN or Inf found")
        features[index] = array.astype(np.float32, copy=False)
        if (index + 1) % 1000 == 0 or index + 1 == n_samples:
            print(f"Materialized {index + 1}/{n_samples} samples")

    del feature_dict
    del key_map
    gc.collect()

    scaler = joblib.load(SCALER_PATH)
    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features not in (None, 6 * 100 * 143):
        raise ValueError(
            f"Scaler n_features_in_={scaler_features}, expected 85800"
        )

    print("Scaling features in bounded-memory chunks...")
    for start in range(0, n_samples, scale_chunk_size):
        end = min(start + scale_chunk_size, n_samples)
        flat = features[start:end].reshape(end - start, -1)
        transformed = scaler.transform(flat)
        features[start:end] = np.asarray(
            transformed.reshape(end - start, 6, 100, 143),
            dtype=np.float32,
        )
        if not np.isfinite(features[start:end]).all():
            raise ValueError(
                f"Scaled features contain NaN/Inf in rows {start}:{end}"
            )
        if end % 1024 == 0 or end == n_samples:
            print(f"Scaled {end}/{n_samples} samples")

    return torch.from_numpy(features), selected_ids, total_available


def build_config(args: argparse.Namespace) -> DFFConfig:
    config = DFFConfig.from_pretrained(str(MODEL_CONFIG_DIR))
    config.update(
        {
            "image_size": (100, 143),
            "patch_size": (1, 143),
            "num_channels": 6,
            "hidden_size": 1024,
            "num_hidden_layers": 12,
            "intermediate_size": 4096,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "decoder_hidden_size": 768,
            "decoder_num_hidden_layers": 8,
            "decoder_num_attention_heads": 12,
            "decoder_intermediate_size": 3072,
            "mask_ratio": args.mask_ratio,
            "norm_pix_loss": True,
            "loss_on_patches": "on_removed_patches",
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
        }
    )
    return config


class DeterministicBatchProvider:
    def __init__(
        self,
        n_samples: int,
        micro_batch_size: int,
        seed: int,
    ) -> None:
        self.indices = np.arange(n_samples, dtype=np.int64)
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.batches_per_epoch = n_samples // micro_batch_size
        if self.batches_per_epoch <= 0:
            raise ValueError("Dataset is smaller than one micro-batch.")
        self._epoch = None
        self._permutation = None

    def batch_indices(self, global_micro_batch: int) -> torch.Tensor:
        epoch = global_micro_batch // self.batches_per_epoch
        batch_in_epoch = global_micro_batch % self.batches_per_epoch
        if epoch != self._epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self._permutation = rng.permutation(self.indices)
            self._epoch = epoch
        start = batch_in_epoch * self.micro_batch_size
        end = start + self.micro_batch_size
        return torch.as_tensor(
            self._permutation[start:end],
            dtype=torch.long,
        )


def finite_gradients(model: torch.nn.Module) -> bool:
    return all(
        torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in module.state_dict().items()
    }


def save_latest_training_state(
    path: Path,
    completed_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    history: list[dict[str, Any]],
) -> None:
    temp_path = path.with_suffix(".tmp")
    state = {
        "completed_step": completed_step,
        "model_state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
        "history": history,
    }
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def strict_encoder_reload(
    encoder_path: Path,
    config: DFFConfig,
) -> dict[str, Any]:
    regression_model = DFFForImageClassification(config)
    state = safe_weights_load(encoder_path)
    result = regression_model.dff.load_state_dict(state, strict=True)
    return {
        "clean": not result.missing_keys and not result.unexpected_keys,
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
    }


def autocast_context(
    device: torch.device,
    precision: str,
):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def main() -> None:
    args = parse_args()
    require_inputs()
    script_path = Path(__file__).resolve()
    resume_path = (
        Path(args.resume_from).resolve() if args.resume_from else None
    )

    if resume_path is None:
        set_reproducible(args.model_seed)
        output_dir = prepare_new_run(args, script_path)
    else:
        output_dir = prepare_resume_run(args, script_path, resume_path)

    append_command(output_dir)
    invocation_started = time.time()

    features, selected_ids, total_available = select_and_materialize_features(
        sample_limit=args.sample_limit,
        data_seed=args.data_seed,
        scale_chunk_size=args.scale_chunk_size,
        output_dir=output_dir,
    )

    config = build_config(args)
    model = DFFForPreTraining(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.precision == "bf16" and device.type != "cuda":
        raise RuntimeError("bf16 mode requires CUDA in this script.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(round(args.total_steps * args.warmup_ratio))
    warmup_steps = max(0, min(args.total_steps - 1, warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.total_steps,
    )

    accumulation_steps = (
        args.effective_batch_size // args.micro_batch_size
    )
    batch_provider = DeterministicBatchProvider(
        n_samples=len(features),
        micro_batch_size=args.micro_batch_size,
        seed=args.data_seed,
    )

    history: list[dict[str, Any]] = []
    completed_step = 0
    if resume_path is not None:
        state = training_state_load(resume_path)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state["rng_state"])
        history = list(state["history"])
        completed_step = int(state["completed_step"])
        print(f"Resumed after update step {completed_step}")

    stop_target = (
        args.stop_after_steps
        if args.stop_after_steps is not None
        else args.total_steps
    )
    if stop_target <= completed_step:
        raise ValueError(
            f"Stop target {stop_target} is not greater than completed step "
            f"{completed_step}"
        )

    latest_state_path = output_dir / "latest_training_state.pt"
    encoder_path = output_dir / "last_encoder_state_dict.bin"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_started = time.time()
    for update_step in range(completed_step + 1, stop_target + 1):
        update_started = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for micro_index in range(accumulation_steps):
            global_micro_batch = (
                (update_step - 1) * accumulation_steps + micro_index
            )
            indices = batch_provider.batch_indices(global_micro_batch)
            batch = features[indices].to(device, non_blocking=True)

            with autocast_context(device, args.precision):
                outputs = model(topological_features=batch)
                raw_loss = outputs.loss

            if raw_loss is None or not torch.isfinite(raw_loss):
                raise RuntimeError(
                    f"Non-finite loss at update {update_step}: {raw_loss}"
                )
            (raw_loss / accumulation_steps).backward()
            micro_losses.append(float(raw_loss.detach().float().cpu()))

        if not finite_gradients(model):
            raise RuntimeError(
                f"Non-finite gradient at update step {update_step}"
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        update_seconds = time.time() - update_started
        mean_loss = float(np.mean(micro_losses))
        row = {
            "update_step": update_step,
            "reconstruction_loss": mean_loss,
            "gradient_norm_before_clip": float(
                grad_norm.detach().float().cpu()
                if isinstance(grad_norm, torch.Tensor)
                else grad_norm
            ),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "update_seconds": update_seconds,
            "effective_samples_per_second": (
                args.effective_batch_size / update_seconds
            ),
            "loss_finite": math.isfinite(mean_loss),
            "gradients_finite": True,
        }
        history.append(row)

        if (
            update_step % args.log_every == 0
            or update_step == 1
            or update_step == stop_target
        ):
            print(
                f"update={update_step}/{args.total_steps} "
                f"loss={mean_loss:.6f} "
                f"grad_norm={row['gradient_norm_before_clip']:.6f} "
                f"lr={row['learning_rate']:.8f} "
                f"sec={update_seconds:.3f}"
            )

        pd.DataFrame(history).to_csv(
            output_dir / "pretraining_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        should_save_state = (
            args.save_state_every > 0
            and (
                update_step % args.save_state_every == 0
                or update_step == stop_target
            )
        )
        if should_save_state:
            save_latest_training_state(
                latest_state_path,
                completed_step=update_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
            )
            print(f"Saved resumable state: {latest_state_path}")

    torch.save(cpu_state_dict(model.dff), encoder_path)
    reload_result = strict_encoder_reload(encoder_path, config)
    if not reload_result["clean"]:
        raise RuntimeError(
            f"Encoder strict reload failed: {reload_result}"
        )

    peak_bytes = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0
    )
    run_finished = stop_target == args.total_steps
    status = "completed" if run_finished else "paused"

    recent_window = history[-min(20, len(history)) :]
    summary = {
        "status": status,
        "scope": (
            "Real-data aligned-Mamba self-supervised pretraining. "
            "No labels or CASF benchmark files were used."
        ),
        "device": str(device),
        "precision": args.precision,
        "available_samples": total_available,
        "selected_samples": len(selected_ids),
        "sample_limit": args.sample_limit,
        "data_seed": args.data_seed,
        "model_seed": args.model_seed,
        "total_steps": args.total_steps,
        "completed_steps": stop_target,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": args.effective_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "mask_ratio": args.mask_ratio,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "initial_reconstruction_loss": history[0][
            "reconstruction_loss"
        ],
        "final_reconstruction_loss": history[-1][
            "reconstruction_loss"
        ],
        "recent_mean_reconstruction_loss": float(
            np.mean(
                [row["reconstruction_loss"] for row in recent_window]
            )
        ),
        "minimum_observed_reconstruction_loss": float(
            min(row["reconstruction_loss"] for row in history)
        ),
        "all_losses_finite": all(
            bool(row["loss_finite"]) for row in history
        ),
        "all_gradients_finite": all(
            bool(row["gradients_finite"]) for row in history
        ),
        "mean_update_seconds": float(
            np.mean([row["update_seconds"] for row in history])
        ),
        "mean_effective_samples_per_second": float(
            np.mean(
                [
                    row["effective_samples_per_second"]
                    for row in history
                ]
            )
        ),
        "peak_gpu_memory_bytes": peak_bytes,
        "peak_gpu_memory_gib": round(peak_bytes / (1024**3), 3),
        "encoder_checkpoint": str(encoder_path),
        "encoder_strict_reload": reload_result,
        "latest_resumable_state": (
            str(latest_state_path)
            if latest_state_path.is_file()
            else None
        ),
        "model_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "encoder_parameters": sum(
            parameter.numel() for parameter in model.dff.parameters()
        ),
        "model_config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "image_size": list(config.image_size),
            "patch_size": list(config.patch_size),
            "num_channels": config.num_channels,
            "decoder_hidden_size": config.decoder_hidden_size,
            "decoder_num_hidden_layers": (
                config.decoder_num_hidden_layers
            ),
            "decoder_num_attention_heads": (
                config.decoder_num_attention_heads
            ),
            "decoder_intermediate_size": (
                config.decoder_intermediate_size
            ),
        },
        "training_elapsed_seconds": round(
            time.time() - training_started, 3
        ),
        "invocation_elapsed_seconds": round(
            time.time() - invocation_started, 3
        ),
    }
    write_json(output_dir / "pretraining_summary.json", summary)
    update_manifest(
        output_dir,
        status=status,
        completed_steps=stop_target,
        selected_samples=len(selected_ids),
        encoder_checkpoint=str(encoder_path),
    )

    print("\n=== Aligned-Mamba real-data pretraining summary ===")
    for key, value in summary.items():
        if key not in {"model_config", "encoder_strict_reload"}:
            print(f"{key}: {value}")
    print(f"encoder_strict_reload: {reload_result}")
    print(f"outputs: {output_dir}")

    if not summary["all_losses_finite"]:
        raise SystemExit("FAIL: non-finite reconstruction loss detected.")
    if not summary["all_gradients_finite"]:
        raise SystemExit("FAIL: non-finite gradient detected.")

    if run_finished:
        print("\nPASS: aligned-Mamba real-data pretraining run completed.")
    else:
        print("\nPASS: aligned-Mamba pretraining paused with valid outputs.")
    print(
        "NEXT: inspect the loss/speed data before increasing the training "
        "length. CASF evaluation is not appropriate at the pretraining stage."
    )


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY.")
        print(
            "Retry with a smaller --micro_batch_size and a new "
            "--output_name. Keep the same effective batch only after the "
            "smoke test has passed."
        )
        raise
