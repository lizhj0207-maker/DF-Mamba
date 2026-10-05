from __future__ import annotations

# Set before importing torch/CUDA libraries.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
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
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset, TensorDataset
from transformers import get_linear_schedule_with_warmup


ORIGINAL_ROOT = Path(r"D:\Capture_Mamba_Clean")
WORK_ROOT = Path(r"D:\Capture_Mamba_Paper_Study")

MODEL_CONFIG_DIR = ORIGINAL_ROOT / "pretrained_model"
FEATURE_PATH = (
    ORIGINAL_ROOT / "data" / "processed" / "v2020_general_train_feat.npy"
)
LABEL_PATH = (
    ORIGINAL_ROOT
    / "data"
    / "labels"
    / "v2020_general_exclude_core_label.csv"
)
SCALER_PATH = (
    ORIGINAL_ROOT / "code_pkg" / "pretrain_data_standard_minmax.sav"
)
SPLIT_PATH = (
    WORK_ROOT
    / "paper_outputs"
    / "02_transformer_baseline"
    / "splits"
    / "pdbbind_internal_val_0p1_splitseed_2026.csv"
)
PRETRAINED_ENCODER = (
    WORK_ROOT
    / "paper_outputs"
    / "03_aligned_mamba"
    / "pretraining"
    / "aligned_mamba_pretrain_formal30000_seed0"
    / "last_encoder_state_dict.bin"
)

sys.path.insert(0, str(WORK_ROOT))

from code_pkg.DF_transformer.configuration_dff import DFFConfig
from code_pkg.DF_transformer.modeling_dff_mamba_ordered import (
    DFFForImageClassification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Protocol-aligned Mamba regression fine-tuning on the exact fixed "
            "PDBbind split previously used by the formal Transformer baseline. "
            "CASF files are never read."
        )
    )
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument(
        "--stop_after_steps",
        type=int,
        default=None,
        help=(
            "Optional controlled stop. Scheduler still uses --total_steps so "
            "the run can resume later."
        ),
    )
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--effective_batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--save_state_every", type=int, default=500)
    parser.add_argument("--scale_chunk_size", type=int, default=64)
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Path to latest_training_state.pt in the same output directory.",
    )
    args = parser.parse_args()

    if args.total_steps <= 0:
        parser.error("--total_steps must be positive")
    if args.stop_after_steps is not None:
        if args.stop_after_steps <= 0:
            parser.error("--stop_after_steps must be positive")
        if args.stop_after_steps > args.total_steps:
            parser.error("--stop_after_steps cannot exceed --total_steps")
    if args.micro_batch_size <= 0 or args.effective_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.effective_batch_size % args.micro_batch_size != 0:
        parser.error(
            "--effective_batch_size must be divisible by --micro_batch_size"
        )
    if args.val_batch_size <= 0:
        parser.error("--val_batch_size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup_ratio must be in [0, 1)")
    if min(args.d_state, args.d_conv, args.expand) <= 0:
        parser.error("Mamba parameters must be positive")
    if args.eval_every <= 0 or args.save_state_every <= 0:
        parser.error("evaluation/save intervals must be positive")
    if args.scale_chunk_size <= 0:
        parser.error("--scale_chunk_size must be positive")
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
        LABEL_PATH,
        SCALER_PATH,
        SPLIT_PATH,
        PRETRAINED_ENCODER,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def fixed_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "total_steps": args.total_steps,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "val_batch_size": args.val_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_clip_norm": args.gradient_clip_norm,
        "model_seed": args.model_seed,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "eval_every": args.eval_every,
        "save_state_every": args.save_state_every,
        "scale_chunk_size": args.scale_chunk_size,
        "output_name": args.output_name,
    }


def output_dir_for(name: str) -> Path:
    return (
        WORK_ROOT
        / "paper_outputs"
        / "03_aligned_mamba"
        / "finetuning"
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

    manifest = {
        "status": "created",
        "scope": (
            "Protocol-aligned Mamba PDBbind fine-tuning using exactly the "
            "Transformer baseline's fixed internal split. CASF files are not read."
        ),
        "script_snapshot": str(snapshot),
        "script_sha256": sha256_file(snapshot),
        "pretrained_encoder": str(PRETRAINED_ENCODER),
        "pretrained_encoder_sha256": sha256_file(PRETRAINED_ENCODER),
        "feature_path": str(FEATURE_PATH),
        "label_path": str(LABEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "split_path": str(SPLIT_PATH),
        "split_sha256": sha256_file(SPLIT_PATH),
        "fixed_arguments": fixed_arguments(args),
        "environment": {
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
        },
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
            f"Resume state belongs to {output_dir}, but --output_name maps to "
            f"{expected}"
        )

    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(script_path) != manifest["script_sha256"]:
        raise RuntimeError(
            "Current script differs from the script snapshot that started the run."
        )
    if fixed_arguments(args) != manifest["fixed_arguments"]:
        raise RuntimeError(
            "Fixed arguments differ from the original run. Only "
            "--resume_from and --stop_after_steps may change."
        )
    if sha256_file(PRETRAINED_ENCODER) != manifest["pretrained_encoder_sha256"]:
        raise RuntimeError("Formal pretrained Mamba encoder has changed.")
    if sha256_file(SPLIT_PATH) != manifest["split_sha256"]:
        raise RuntimeError("Fixed PDBbind split file has changed.")
    return output_dir


def append_command(output_dir: Path) -> None:
    command = " ".join(shlex.quote(part) for part in sys.argv)
    with (output_dir / "command_history.txt").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {command}\n")


def update_manifest(output_dir: Path, **updates: Any) -> None:
    path = output_dir / "run_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(updates)
    write_json(path, value)


def load_labels_and_split() -> pd.DataFrame:
    raw = pd.read_csv(LABEL_PATH, header=0, index_col=0)
    if raw.shape[1] < 1:
        raise ValueError("Label CSV has no numeric target column.")
    values = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    if values.isna().any():
        raise ValueError("Label CSV contains missing/non-numeric target values.")

    labels = pd.DataFrame(
        {
            "normalized_id": [normalize_id(value) for value in raw.index],
            "label": values.astype(float).to_numpy(),
        }
    )
    if labels["normalized_id"].duplicated().any():
        raise ValueError("Duplicate normalized IDs in the label file.")

    split = pd.read_csv(SPLIT_PATH)
    required = {"normalized_id", "split", "label"}
    if not required.issubset(split.columns):
        raise ValueError(f"Fixed split lacks required columns: {required}")
    if set(split["normalized_id"]) != set(labels["normalized_id"]):
        raise ValueError("Fixed split IDs do not match current PDBbind labels.")
    if set(split["split"]) != {"train", "validation"}:
        raise ValueError("Fixed split must contain train and validation.")
    if (split["split"] == "train").sum() != 17013:
        raise ValueError("Fixed split no longer contains 17,013 training samples.")
    if (split["split"] == "validation").sum() != 1891:
        raise ValueError("Fixed split no longer contains 1,891 validation samples.")

    merged = labels.merge(
        split[["normalized_id", "split"]],
        on="normalized_id",
        how="left",
        validate="one_to_one",
    )
    if merged["split"].isna().any():
        raise RuntimeError("Some labels are missing a split assignment.")
    return merged


def materialize_features(
    table: pd.DataFrame,
    scale_chunk_size: int,
) -> torch.Tensor:
    print(f"Loading audited fine-tuning feature dictionary: {FEATURE_PATH}")
    feature_dict = np.load(FEATURE_PATH, allow_pickle=True).item()
    key_map: dict[str, object] = {}
    for raw_key in feature_dict:
        normalized = normalize_id(raw_key)
        if normalized in key_map:
            raise ValueError(f"Duplicate normalized feature ID: {normalized}")
        key_map[normalized] = raw_key

    missing = [
        sample_id
        for sample_id in table["normalized_id"]
        if sample_id not in key_map
    ]
    if missing:
        raise KeyError(f"Missing fine-tuning features: {missing[:10]}")

    n_samples = len(table)
    features = np.empty((n_samples, 6, 100, 143), dtype=np.float32)
    for index, sample_id in enumerate(table["normalized_id"]):
        array = np.asarray(feature_dict[key_map[sample_id]])
        if array.shape != (6, 100, 143):
            raise ValueError(f"{sample_id}: unexpected feature shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{sample_id}: NaN or Inf in features")
        features[index] = array.astype(np.float32, copy=False)
        if (index + 1) % 1000 == 0 or index + 1 == n_samples:
            print(f"Materialized {index + 1}/{n_samples} samples")

    del feature_dict, key_map
    gc.collect()

    scaler = joblib.load(SCALER_PATH)
    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features not in (None, 6 * 100 * 143):
        raise ValueError(
            f"Scaler n_features_in_={scaler_features}, expected 85800."
        )

    print("Scaling features...")
    for start in range(0, n_samples, scale_chunk_size):
        end = min(start + scale_chunk_size, n_samples)
        flat = features[start:end].reshape(end - start, -1)
        transformed = scaler.transform(flat)
        features[start:end] = np.asarray(
            transformed.reshape(end - start, 6, 100, 143),
            dtype=np.float32,
        )
        if not np.isfinite(features[start:end]).all():
            raise ValueError("Scaled features contain NaN or Inf.")
        if end % 1024 == 0 or end == n_samples:
            print(f"Scaled {end}/{n_samples} samples")
    return torch.from_numpy(features)


def build_model(args: argparse.Namespace) -> tuple[DFFForImageClassification, DFFConfig]:
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
            "mask_ratio": 0.0,
            "num_labels": 1,
            "pooler_type": "cls_token",
            "cls_token_position": "last",
            "specify_loss_fct": "mse",
            "d_state": args.d_state,
            "d_conv": args.d_conv,
            "expand": args.expand,
        }
    )
    model = DFFForImageClassification(config)

    encoder_state = safe_weights_load(PRETRAINED_ENCODER)
    result = model.dff.load_state_dict(encoder_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Formal Mamba encoder load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model, config


class DeterministicMicroBatchProvider:
    def __init__(
        self,
        train_indices: list[int],
        micro_batch_size: int,
        seed: int,
    ) -> None:
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.batches_per_epoch = len(self.train_indices) // micro_batch_size
        if self.batches_per_epoch <= 0:
            raise ValueError("Training set smaller than one micro-batch.")
        self._cached_epoch = None
        self._cached_permutation = None

    def batch_indices(self, global_micro_batch: int) -> torch.Tensor:
        epoch = global_micro_batch // self.batches_per_epoch
        batch_in_epoch = global_micro_batch % self.batches_per_epoch
        if epoch != self._cached_epoch:
            rng = np.random.default_rng(self.seed + epoch)
            self._cached_permutation = rng.permutation(self.train_indices)
            self._cached_epoch = epoch
        start = batch_in_epoch * self.micro_batch_size
        end = start + self.micro_batch_size
        return torch.as_tensor(
            self._cached_permutation[start:end],
            dtype=torch.long,
        )


def evaluate(
    model: DFFForImageClassification,
    loader: DataLoader,
    device: torch.device,
    ids: list[str],
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    predictions: list[float] = []
    truths: list[float] = []
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            outputs = model(topological_features=features)
            predictions.extend(
                outputs.logits.detach().float().cpu().view(-1).tolist()
            )
            truths.extend(labels.float().cpu().view(-1).tolist())

    pred = np.asarray(predictions, dtype=np.float64)
    true = np.asarray(truths, dtype=np.float64)
    if len(pred) != len(ids):
        raise RuntimeError("Validation prediction count mismatch.")
    metrics = {
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "mae": float(np.mean(np.abs(pred - true))),
        "pcc": float(pearsonr(true, pred).statistic),
        "n": int(len(true)),
    }
    frame = pd.DataFrame(
        {
            "normalized_id": ids,
            "true_ba": true,
            "predicted_ba": pred,
            "error": pred - true,
        }
    )
    return metrics, frame


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


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_latest_state(
    path: Path,
    completed_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_rmse: float,
    best_step: int,
    history: list[dict[str, Any]],
    validation_history: list[dict[str, Any]],
) -> None:
    temp = path.with_suffix(".tmp")
    torch.save(
        {
            "completed_step": completed_step,
            "model_state_dict": cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_rmse": best_rmse,
            "best_step": best_step,
            "rng_state": capture_rng_state(),
            "history": history,
            "validation_history": validation_history,
        },
        temp,
    )
    os.replace(temp, path)


def strict_reload(
    checkpoint: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fresh, _ = build_model(args)
    state = safe_weights_load(checkpoint)
    result = fresh.load_state_dict(state, strict=True)
    return {
        "clean": not result.missing_keys and not result.unexpected_keys,
        "missing_keys": result.missing_keys,
        "unexpected_keys": result.unexpected_keys,
    }


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

    table = load_labels_and_split()
    features = materialize_features(table, args.scale_chunk_size)
    labels = torch.from_numpy(
        table["label"].to_numpy(dtype=np.float32)
    )

    train_indices = table.index[table["split"] == "train"].tolist()
    val_indices = table.index[table["split"] == "validation"].tolist()
    val_ids = table.loc[val_indices, "normalized_id"].tolist()

    dataset = TensorDataset(features, labels)
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    accumulation_steps = (
        args.effective_batch_size // args.micro_batch_size
    )
    batch_provider = DeterministicMicroBatchProvider(
        train_indices=train_indices,
        micro_batch_size=args.micro_batch_size,
        seed=args.model_seed,
    )

    model, config = build_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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

    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    best_rmse = float("inf")
    best_step = -1
    completed_step = 0

    if resume_path is not None:
        state = training_state_load(resume_path)
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        restore_rng_state(state["rng_state"])
        best_rmse = float(state["best_rmse"])
        best_step = int(state["best_step"])
        history = list(state["history"])
        validation_history = list(state["validation_history"])
        completed_step = int(state["completed_step"])
        print(f"Resumed after update step {completed_step}")

    stop_target = (
        args.stop_after_steps
        if args.stop_after_steps is not None
        else args.total_steps
    )
    if stop_target <= completed_step:
        raise ValueError("Stop target must exceed completed step.")

    best_checkpoint = output_dir / "best_model_state_dict.bin"
    last_checkpoint = output_dir / "last_model_state_dict.bin"
    latest_state = output_dir / "latest_training_state.pt"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    training_started = time.time()
    for update_step in range(completed_step + 1, stop_target + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []

        for micro_index in range(accumulation_steps):
            global_micro_batch = (
                (update_step - 1) * accumulation_steps + micro_index
            )
            indices = batch_provider.batch_indices(global_micro_batch)
            batch_features = features[indices].to(device, non_blocking=True)
            batch_labels = labels[indices].to(device, non_blocking=True)

            outputs = model(
                topological_features=batch_features,
                labels=batch_labels,
            )
            raw_loss = outputs.loss
            if raw_loss is None or not torch.isfinite(raw_loss):
                raise RuntimeError(
                    f"Non-finite loss at update {update_step}: {raw_loss}"
                )
            (raw_loss / accumulation_steps).backward()
            micro_losses.append(float(raw_loss.detach().cpu()))

        if not finite_gradients(model):
            raise RuntimeError(
                f"Non-finite gradient at update {update_step}"
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()

        mean_loss = float(np.mean(micro_losses))
        history.append(
            {
                "update_step": update_step,
                "mean_microbatch_loss": mean_loss,
                "gradient_norm_before_clip": float(
                    grad_norm.detach().cpu()
                    if isinstance(grad_norm, torch.Tensor)
                    else grad_norm
                ),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "loss_finite": math.isfinite(mean_loss),
                "gradients_finite": True,
            }
        )
        print(
            f"update={update_step}/{args.total_steps} "
            f"loss={mean_loss:.6f} "
            f"grad_norm={history[-1]['gradient_norm_before_clip']:.6f} "
            f"lr={history[-1]['learning_rate']:.8f}"
        )

        if (
            update_step % args.eval_every == 0
            or update_step == args.total_steps
        ):
            metrics, predictions = evaluate(
                model, val_loader, device, val_ids
            )
            validation_history.append(
                {"update_step": update_step, **metrics}
            )
            predictions.to_csv(
                output_dir
                / f"validation_predictions_step_{update_step}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"validation step={update_step}: n={metrics['n']} "
                f"rmse={metrics['rmse']:.6f} "
                f"mae={metrics['mae']:.6f} pcc={metrics['pcc']:.6f}"
            )
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_step = update_step
                torch.save(cpu_state_dict(model), best_checkpoint)
                predictions.to_csv(
                    output_dir / "best_validation_predictions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        pd.DataFrame(history).to_csv(
            output_dir / "training_history.csv",
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame(validation_history).to_csv(
            output_dir / "validation_history.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if (
            update_step % args.save_state_every == 0
            or update_step == stop_target
        ):
            save_latest_state(
                latest_state,
                completed_step=update_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_rmse=best_rmse,
                best_step=best_step,
                history=history,
                validation_history=validation_history,
            )
            print(f"Saved resumable state: {latest_state}")

    torch.save(cpu_state_dict(model), last_checkpoint)

    strict_results = {
        "last": strict_reload(last_checkpoint, args),
    }
    if best_checkpoint.is_file():
        strict_results["best"] = strict_reload(best_checkpoint, args)

    peak_bytes = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available()
        else 0
    )
    finished = stop_target == args.total_steps
    summary = {
        "status": "completed" if finished else "paused",
        "scope": (
            "Aligned-Mamba regression fine-tuning on the exact fixed "
            "Transformer PDBbind train/validation split; CASF not used."
        ),
        "device": str(device),
        "total_samples": int(len(table)),
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "model_seed": args.model_seed,
        "total_steps": args.total_steps,
        "completed_steps": stop_target,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": args.effective_batch_size,
        "val_batch_size": args.val_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "d_state": args.d_state,
        "d_conv": args.d_conv,
        "expand": args.expand,
        "initial_training_loss": history[0]["mean_microbatch_loss"],
        "final_training_loss": history[-1]["mean_microbatch_loss"],
        "all_losses_finite": all(
            bool(row["loss_finite"]) for row in history
        ),
        "all_gradients_finite": all(
            bool(row["gradients_finite"]) for row in history
        ),
        "best_validation_rmse": (
            best_rmse if math.isfinite(best_rmse) else None
        ),
        "best_step": best_step,
        "peak_gpu_memory_bytes": peak_bytes,
        "peak_gpu_memory_gib": round(peak_bytes / (1024**3), 3),
        "best_checkpoint": (
            str(best_checkpoint) if best_checkpoint.is_file() else None
        ),
        "last_checkpoint": str(last_checkpoint),
        "latest_resumable_state": str(latest_state),
        "strict_reload": strict_results,
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
            "pooler_type": config.pooler_type,
            "cls_token_position": getattr(config, "cls_token_position", "first"),
            "specify_loss_fct": config.specify_loss_fct,
        },
        "training_elapsed_seconds_this_invocation": round(
            time.time() - training_started, 3
        ),
        "invocation_elapsed_seconds": round(
            time.time() - invocation_started, 3
        ),
    }
    write_json(output_dir / "finetuning_summary.json", summary)
    update_manifest(
        output_dir,
        status=summary["status"],
        completed_steps=stop_target,
        best_step=best_step,
        best_validation_rmse=summary["best_validation_rmse"],
    )

    print("\n=== Aligned-Mamba fine-tuning summary ===")
    for key, value in summary.items():
        if key not in {"model_config", "strict_reload"}:
            print(f"{key}: {value}")
    print(f"strict_reload: {strict_results}")
    print(f"outputs: {output_dir}")

    if not summary["all_losses_finite"] or not summary["all_gradients_finite"]:
        raise SystemExit("FAIL: non-finite loss or gradient detected.")
    if any(
        result.get("clean") is False
        for result in strict_results.values()
    ):
        raise SystemExit("FAIL: strict checkpoint reload failed.")

    if finished:
        print("\nPASS: formal aligned-Mamba fine-tuning completed.")
        print(
            "NEXT: freeze the internally selected best checkpoint and run "
            "CASF-2007/2013/2016 independent evaluation."
        )
    else:
        print("\nPASS: aligned-Mamba fine-tuning paused with a valid state.")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA OUT OF MEMORY.")
        print(
            "Retry with a smaller micro batch and a new output name. "
            "Keep effective batch 32 for the formal comparison."
        )
        raise
