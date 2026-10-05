from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from transformers import get_linear_schedule_with_warmup


ORIGINAL_ROOT = Path(r"D:\Capture_Mamba_Clean")
WORK_ROOT = Path(r"D:\Capture_Mamba_Paper_Study")

FEATURE_PATH = (
    ORIGINAL_ROOT
    / "data"
    / "processed"
    / "SARS_Features"
    / "SARS_CoV2_CoV_features_203.npy"
)
LABEL_PATH = (
    ORIGINAL_ROOT
    / "data"
    / "labels"
    / "SARS_CoV2_CoV_labels.csv"
)
SCALER_PATH = (
    ORIGINAL_ROOT
    / "code_pkg"
    / "pretrain_data_standard_minmax.sav"
)
MODEL_CONFIG_DIR = ORIGINAL_ROOT / "pretrained_model"

BASE_CHECKPOINT = (
    WORK_ROOT
    / "paper_outputs"
    / "03_aligned_mamba"
    / "finetuning"
    / "aligned_mamba_cls_last_formal10000_seed0"
    / "best_model_state_dict.bin"
)

sys.path.insert(0, str(WORK_ROOT))

from code_pkg.DF_transformer.configuration_dff import DFFConfig
from code_pkg.DF_transformer.modeling_dff_mamba_ordered import (
    DFFForImageClassification,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Five-fold second-stage regression fine-tuning of the frozen "
            "PDBbind-trained ordered CLS-last Aligned-Mamba on the currently "
            "available matched local small dataset. No data repair occurs."
        )
    )
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=8e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--split_seed", type=int, default=2026)
    p.add_argument("--model_seed", type=int, default=0)
    p.add_argument(
        "--only_fold",
        type=int,
        default=0,
        help="0 runs all 5 folds; 1..5 runs only that fold (for smoke tests).",
    )
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--output_name", required=True)
    args = p.parse_args()

    if args.steps <= 0:
        p.error("--steps must be positive")
    if args.batch_size <= 0:
        p.error("--batch_size must be positive")
    if args.learning_rate <= 0:
        p.error("--learning_rate must be positive")
    if not 0 <= args.warmup_ratio < 1:
        p.error("--warmup_ratio must be in [0,1)")
    if args.only_fold not in (0, 1, 2, 3, 4, 5):
        p.error("--only_fold must be 0 or 1..5")
    if not args.output_name.strip():
        p.error("--output_name cannot be empty")
    return args


def normalize_id(value: object) -> str:
    return str(value).strip().lower()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def require_files() -> None:
    for path in (
        FEATURE_PATH,
        LABEL_PATH,
        SCALER_PATH,
        MODEL_CONFIG_DIR / "config.json",
        BASE_CHECKPOINT,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def read_labels() -> pd.DataFrame:
    raw = pd.read_csv(LABEL_PATH, header=0, index_col=0)
    if raw.shape[1] < 1:
        raise ValueError("Label CSV has no target column.")

    # Use the first column that is fully numeric after coercion.
    chosen = None
    values = None
    for column in raw.columns:
        candidate = pd.to_numeric(raw[column], errors="coerce")
        if candidate.notna().all():
            chosen = str(column)
            values = candidate.astype(float)
            break
    if chosen is None or values is None:
        raise ValueError(
            f"Could not identify a fully numeric regression target column. "
            f"Columns: {list(raw.columns)}"
        )

    table = pd.DataFrame(
        {
            "sample_id": [str(v).strip() for v in raw.index],
            "normalized_id": [normalize_id(v) for v in raw.index],
            "true_ba": values.to_numpy(dtype=float),
        }
    )
    if table["normalized_id"].duplicated().any():
        raise ValueError("Duplicate normalized IDs in labels.")
    print(f"Label target column: {chosen}")
    return table


def load_and_match() -> tuple[pd.DataFrame, torch.Tensor, dict[str, Any]]:
    labels = read_labels()
    feature_dict = np.load(FEATURE_PATH, allow_pickle=True).item()

    key_map: dict[str, object] = {}
    for raw_key in feature_dict:
        normalized = normalize_id(raw_key)
        if normalized in key_map:
            raise ValueError(f"Duplicate normalized feature key: {normalized}")
        key_map[normalized] = raw_key

    feature_ids = set(key_map)
    label_ids = set(labels["normalized_id"])
    matched_ids = sorted(feature_ids & label_ids)
    extra_feature_ids = sorted(feature_ids - label_ids)
    missing_feature_ids = sorted(label_ids - feature_ids)

    audit = {
        "feature_records": len(feature_ids),
        "label_records": len(label_ids),
        "matched_records": len(matched_ids),
        "extra_feature_ids": extra_feature_ids,
        "missing_feature_ids": missing_feature_ids,
        "feature_path": str(FEATURE_PATH),
        "label_path": str(LABEL_PATH),
    }

    print("=== Local data audit ===")
    print(f"feature_records: {audit['feature_records']}")
    print(f"label_records: {audit['label_records']}")
    print(f"matched_records: {audit['matched_records']}")
    print(f"extra_feature_ids: {len(extra_feature_ids)}")
    print(f"missing_feature_ids: {len(missing_feature_ids)}")

    # User explicitly chose not to repair the dataset.
    if len(matched_ids) != 163:
        raise RuntimeError(
            f"Expected exactly 163 matched records from the completed audit, "
            f"but found {len(matched_ids)}. No automatic repair will be attempted."
        )

    label_map = labels.set_index("normalized_id")
    rows = []
    features = np.empty((len(matched_ids), 6, 100, 143), dtype=np.float32)

    for i, sample_id in enumerate(matched_ids):
        arr = np.asarray(feature_dict[key_map[sample_id]])
        if arr.shape != (6, 100, 143):
            raise ValueError(f"{sample_id}: unexpected feature shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{sample_id}: feature contains NaN/Inf")
        features[i] = arr.astype(np.float32, copy=False)
        row = label_map.loc[sample_id]
        rows.append(
            {
                "sample_id": str(row["sample_id"]),
                "normalized_id": sample_id,
                "true_ba": float(row["true_ba"]),
            }
        )

    del feature_dict
    gc.collect()

    scaler = joblib.load(SCALER_PATH)
    n = len(features)
    flat = features.reshape(n, -1)
    scaled = scaler.transform(flat)
    features = np.asarray(
        scaled.reshape(n, 6, 100, 143),
        dtype=np.float32,
    )
    if not np.isfinite(features).all():
        raise ValueError("Scaled features contain NaN/Inf.")

    return pd.DataFrame(rows), torch.from_numpy(features), audit


def build_config() -> DFFConfig:
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
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
        }
    )
    return config


def fresh_model(device: torch.device) -> DFFForImageClassification:
    config = build_config()
    model = DFFForImageClassification(config)
    state = safe_load(BASE_CHECKPOINT)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Base checkpoint strict load failed: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return model.to(device)


class BatchProvider:
    def __init__(
        self,
        train_indices: np.ndarray,
        batch_size: int,
        seed: int,
    ):
        self.indices = np.asarray(train_indices, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.batches_per_epoch = max(1, len(self.indices) // self.batch_size)
        self.cached_epoch = None
        self.permutation = None

    def get(self, global_batch: int) -> torch.Tensor:
        epoch = global_batch // self.batches_per_epoch
        batch_in_epoch = global_batch % self.batches_per_epoch
        if epoch != self.cached_epoch:
            self.permutation = np.random.default_rng(
                self.seed + epoch
            ).permutation(self.indices)
            self.cached_epoch = epoch

        start = batch_in_epoch * self.batch_size
        end = start + self.batch_size
        if end <= len(self.permutation):
            batch = self.permutation[start:end]
        else:
            # Wrap deterministically if the final slot is incomplete.
            first = self.permutation[start:]
            needed = self.batch_size - len(first)
            batch = np.concatenate([first, self.permutation[:needed]])
        return torch.as_tensor(batch, dtype=torch.long)


def metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    pred_std = float(np.std(pred))
    pcc = (
        float(pearsonr(true, pred).statistic)
        if pred_std > 1e-12
        else float("nan")
    )
    return {
        "rmse": rmse,
        "mae": mae,
        "pcc": pcc,
        "prediction_std": pred_std,
    }


def evaluate(
    model: DFFForImageClassification,
    features: torch.Tensor,
    table: pd.DataFrame,
    test_indices: np.ndarray,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    preds = []
    truths = []
    batch_size = 16
    with torch.no_grad():
        for start in range(0, len(test_indices), batch_size):
            idx = test_indices[start:start + batch_size]
            x = features[idx].to(device)
            out = model(topological_features=x)
            preds.extend(
                out.logits.detach().float().cpu().view(-1).tolist()
            )
            truths.extend(table.iloc[idx]["true_ba"].astype(float).tolist())

    true = np.asarray(truths, dtype=np.float64)
    pred = np.asarray(preds, dtype=np.float64)
    m = metrics(true, pred)
    frame = pd.DataFrame(
        {
            "sample_id": table.iloc[test_indices]["sample_id"].tolist(),
            "normalized_id": table.iloc[test_indices]["normalized_id"].tolist(),
            "true_ba": true,
            "predicted_ba": pred,
            "error": pred - true,
            "absolute_error": np.abs(pred - true),
        }
    )
    return m, frame


def run_fold(
    fold_number: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    features: torch.Tensor,
    table: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_number}"
    fold_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.model_seed + fold_number)
    model = fresh_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    warmup_steps = int(round(args.steps * args.warmup_ratio))
    warmup_steps = max(0, min(args.steps - 1, warmup_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.steps,
    )

    provider = BatchProvider(
        train_indices=train_indices,
        batch_size=args.batch_size,
        seed=args.split_seed + fold_number,
    )

    history = []
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        idx = provider.get(step - 1)
        x = features[idx].to(device)
        y = torch.as_tensor(
            table.iloc[idx.numpy()]["true_ba"].to_numpy(dtype=np.float32),
            device=device,
        )

        out = model(topological_features=x, labels=y)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(
                f"Fold {fold_number}: non-finite loss at step {step}"
            )
        loss.backward()

        if not all(
            torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.grad is not None
        ):
            raise RuntimeError(
                f"Fold {fold_number}: non-finite gradient at step {step}"
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm_before_clip": float(grad_norm.detach().cpu()),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)

        if (
            step == 1
            or step % args.log_every == 0
            or step == args.steps
        ):
            print(
                f"fold={fold_number} step={step}/{args.steps} "
                f"loss={row['loss']:.6f} "
                f"grad_norm={row['grad_norm_before_clip']:.6f} "
                f"lr={row['learning_rate']:.8f}"
            )

    pd.DataFrame(history).to_csv(
        fold_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fold_metrics, predictions = evaluate(
        model, features, table, test_indices, device
    )
    predictions.insert(1, "fold", fold_number)
    predictions.to_csv(
        fold_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_record = {
        "fold": fold_number,
        "n_train": int(len(train_indices)),
        "n_test": int(len(test_indices)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        **fold_metrics,
        "elapsed_seconds": round(time.time() - start_time, 3),
    }
    write_json(fold_dir / "metrics.json", metric_record)

    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        fold_dir / "final_model_state_dict.bin",
    )

    print(
        f"FOLD {fold_number} RESULT: "
        f"n={metric_record['n_test']} "
        f"RMSE={metric_record['rmse']:.6f} "
        f"MAE={metric_record['mae']:.6f} "
        f"PCC={metric_record['pcc']:.6f}"
    )

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    return metric_record


def main() -> None:
    args = parse_args()
    require_files()

    output_dir = (
        WORK_ROOT
        / "paper_outputs"
        / "04_mpro_5fold"
        / args.output_name
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    table, features, audit = load_and_match()
    write_json(output_dir / "data_audit.json", audit)

    # Verify formal base checkpoint before any fold starts.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = fresh_model(device)
    del probe
    torch.cuda.empty_cache()
    print("Base checkpoint strict=True load: OK")

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=args.split_seed,
    )

    fold_assignments = np.zeros(len(table), dtype=int)
    split_records = []
    for fold_number, (train_idx, test_idx) in enumerate(
        kfold.split(np.arange(len(table))),
        start=1,
    ):
        fold_assignments[test_idx] = fold_number
        split_records.append((fold_number, train_idx, test_idx))

    assignment_frame = table.copy()
    assignment_frame["fold"] = fold_assignments
    assignment_frame.to_csv(
        output_dir / "fold_assignments.csv",
        index=False,
        encoding="utf-8-sig",
    )

    counts = assignment_frame["fold"].value_counts().sort_index().to_dict()
    print(f"Fold sizes: {counts}")

    if sorted(assignment_frame["fold"].tolist()).count(0) != 0:
        raise RuntimeError("Fold assignment contains unassigned samples.")
    if assignment_frame["normalized_id"].duplicated().any():
        raise RuntimeError("Duplicate sample IDs in fold assignments.")

    folds_to_run = (
        [args.only_fold] if args.only_fold else [1, 2, 3, 4, 5]
    )

    fold_results = []
    for fold_number, train_idx, test_idx in split_records:
        if fold_number not in folds_to_run:
            continue
        fold_results.append(
            run_fold(
                fold_number=fold_number,
                train_indices=train_idx,
                test_indices=test_idx,
                features=features,
                table=table,
                args=args,
                output_dir=output_dir,
                device=device,
            )
        )

    metrics_frame = pd.DataFrame(fold_results)
    metrics_frame.to_csv(
        output_dir / "mpro_5fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    all_predictions = []
    for fold_number in folds_to_run:
        path = output_dir / f"fold_{fold_number}" / "predictions.csv"
        if path.is_file():
            all_predictions.append(pd.read_csv(path))
    all_predictions_frame = pd.concat(all_predictions, ignore_index=True)
    all_predictions_frame.to_csv(
        output_dir / "all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    true = all_predictions_frame["true_ba"].to_numpy(dtype=float)
    pred = all_predictions_frame["predicted_ba"].to_numpy(dtype=float)
    pooled = metrics(true, pred)

    summary = {
        "status": "completed",
        "scope": (
            "Five-fold second-stage fine-tuning using only the 163 locally "
            "matched feature/label records. No data repair or label recovery."
        ),
        "formal_full_5fold": args.only_fold == 0 and args.steps == 4000,
        "matched_samples": int(len(table)),
        "folds_run": folds_to_run,
        "fold_sizes": counts,
        "steps_per_fold": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
        "mean_rmse": float(metrics_frame["rmse"].mean()),
        "std_rmse": float(metrics_frame["rmse"].std(ddof=1))
        if len(metrics_frame) > 1 else None,
        "mean_mae": float(metrics_frame["mae"].mean()),
        "std_mae": float(metrics_frame["mae"].std(ddof=1))
        if len(metrics_frame) > 1 else None,
        "mean_pcc": float(metrics_frame["pcc"].mean()),
        "std_pcc": float(metrics_frame["pcc"].std(ddof=1))
        if len(metrics_frame) > 1 else None,
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "pooled_pcc": pooled["pcc"],
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": sha256_file(BASE_CHECKPOINT),
        "feature_path": str(FEATURE_PATH),
        "label_path": str(LABEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }
    write_json(output_dir / "mpro_5fold_summary.json", summary)

    manifest = {
        "status": "completed",
        "model": "ordered CLS-last Aligned-Mamba",
        "model_config": {
            "hidden_size": 1024,
            "layers": 12,
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "pooler_type": "cls_token",
            "cls_token_position": "last",
        },
        "data_policy": (
            "Use existing local matched records only; no data repair."
        ),
        "arguments": vars(args),
        "summary": summary,
    }
    write_json(output_dir / "run_manifest.json", manifest)

    print("\n=== 5-fold summary ===")
    print(metrics_frame.to_string(index=False))
    print(f"\nmean RMSE: {summary['mean_rmse']:.6f}")
    print(f"mean MAE : {summary['mean_mae']:.6f}")
    print(f"mean PCC : {summary['mean_pcc']:.6f}")
    print(f"pooled RMSE: {summary['pooled_rmse']:.6f}")
    print(f"pooled MAE : {summary['pooled_mae']:.6f}")
    print(f"pooled PCC : {summary['pooled_pcc']:.6f}")
    print(f"\nOutputs: {output_dir}")
    print("PASS: local matched-sample 5-fold regression run completed.")


if __name__ == "__main__":
    main()
