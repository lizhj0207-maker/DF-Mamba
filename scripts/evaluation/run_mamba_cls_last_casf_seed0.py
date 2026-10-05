from __future__ import annotations

import argparse
import hashlib
import json
import platform
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
from torch.utils.data import DataLoader, TensorDataset


ORIGINAL_ROOT = Path(r"D:\Capture_Mamba_Clean")
WORK_ROOT = Path(r"D:\Capture_Mamba_Paper_Study")

MODEL_DIR = ORIGINAL_ROOT / "pretrained_model"
SCALER_PATH = (
    ORIGINAL_ROOT / "code_pkg" / "pretrain_data_standard_minmax.sav"
)

FORMAL_RUN_DIR = (
    WORK_ROOT
    / "paper_outputs"
    / "03_aligned_mamba"
    / "finetuning"
    / "aligned_mamba_cls_last_formal10000_seed0"
)
DEFAULT_CHECKPOINT = FORMAL_RUN_DIR / "best_model_state_dict.bin"
FORMAL_SUMMARY = FORMAL_RUN_DIR / "finetuning_summary.json"

DATASETS = {
    "CASF-2007": {
        "features": ORIGINAL_ROOT
        / "data"
        / "processed"
        / "CASF_2007_valid_feat.npy",
        "labels": ORIGINAL_ROOT
        / "data"
        / "labels"
        / "CASF2007_core_test_label.csv",
    },
    "CASF-2013": {
        "features": ORIGINAL_ROOT
        / "data"
        / "processed"
        / "CASF_2013_valid_feat.npy",
        "labels": ORIGINAL_ROOT
        / "data"
        / "labels"
        / "CASF2013_core_test_label.csv",
    },
    "CASF-2016": {
        "features": ORIGINAL_ROOT
        / "data"
        / "processed"
        / "CASF_2016_valid_feat.npy",
        "labels": ORIGINAL_ROOT
        / "data"
        / "labels"
        / "CASF2016_core_test_label.csv",
    },
}

sys.path.insert(0, str(WORK_ROOT))

from code_pkg.DF_transformer.configuration_dff import DFFConfig
from code_pkg.DF_transformer.modeling_dff_mamba_ordered import (
    DFFForImageClassification,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent CASF evaluation of the frozen ordered CLS-last "
            "Aligned-Mamba seed-0 checkpoint. No training, tuning, or "
            "checkpoint selection occurs."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--output_name",
        type=str,
        default="aligned_mamba_cls_last_seed0_casf_independent",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
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


def safe_weights_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def validate_inputs(checkpoint: Path) -> dict[str, Any]:
    required = [
        MODEL_DIR / "config.json",
        SCALER_PATH,
        checkpoint,
        FORMAL_SUMMARY,
    ]
    for dataset in DATASETS.values():
        required.extend([dataset["features"], dataset["labels"]])
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    formal_summary = json.loads(
        FORMAL_SUMMARY.read_text(encoding="utf-8")
    )
    if formal_summary.get("status") != "completed":
        raise RuntimeError("Formal Mamba fine-tuning is not marked completed.")
    if formal_summary.get("completed_steps") != 10000:
        raise RuntimeError("Formal Mamba fine-tuning did not complete 10,000 steps.")
    if formal_summary.get("model_config", {}).get("cls_token_position") != "last":
        raise RuntimeError(
            "Formal summary does not identify the ordered CLS-last model."
        )
    if formal_summary.get("model_config", {}).get("pooler_type") != "cls_token":
        raise RuntimeError("Formal summary does not use CLS-token pooling.")
    return formal_summary


def build_model(
    checkpoint: Path,
) -> tuple[DFFForImageClassification, DFFConfig]:
    config = DFFConfig.from_pretrained(str(MODEL_DIR))
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
    model = DFFForImageClassification(config)
    state = safe_weights_load(checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f"Expected state_dict, got {type(state)}")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint load failed: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    return model, config


def load_label_table(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=0, index_col=0)
    if raw.shape[1] < 1:
        raise ValueError(f"No label value column in {path}")
    values = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{path}: {int(values.isna().sum())} missing/non-numeric labels"
        )
    table = pd.DataFrame(
        {
            "sample_id": [str(value).strip() for value in raw.index],
            "normalized_id": [normalize_id(value) for value in raw.index],
            "true_ba": values.astype(float).to_numpy(),
        }
    )
    if table["normalized_id"].duplicated().any():
        raise ValueError(f"Duplicate normalized label IDs in {path}")
    return table


def prepare_dataset(
    feature_path: Path,
    label_path: Path,
    scaler: Any,
) -> tuple[TensorDataset, pd.DataFrame]:
    labels = load_label_table(label_path)
    feature_dict = np.load(feature_path, allow_pickle=True).item()

    key_map: dict[str, object] = {}
    for raw_key in feature_dict:
        normalized = normalize_id(raw_key)
        if normalized in key_map:
            raise ValueError(
                f"Duplicate normalized feature key {normalized} in {feature_path}"
            )
        key_map[normalized] = raw_key

    missing = [
        sample_id
        for sample_id in labels["normalized_id"]
        if sample_id not in key_map
    ]
    extra = sorted(set(key_map) - set(labels["normalized_id"]))
    if missing:
        raise KeyError(
            f"{feature_path}: {len(missing)} labeled samples lack features"
        )
    if extra:
        raise RuntimeError(
            f"{feature_path}: {len(extra)} extra feature records were found; "
            "the audited CASF benchmark should be exactly aligned."
        )

    features = np.stack(
        [
            np.asarray(feature_dict[key_map[sample_id]])
            for sample_id in labels["normalized_id"]
        ],
        axis=0,
    )
    if features.shape[1:] != (6, 100, 143):
        raise ValueError(f"Unexpected feature array shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"{feature_path}: NaN or Inf detected")

    n = len(features)
    scaled = scaler.transform(features.reshape(n, -1))
    scaled = np.asarray(
        scaled.reshape(n, 6, 100, 143),
        dtype=np.float32,
    )
    if not np.isfinite(scaled).all():
        raise ValueError(f"{feature_path}: scaled features contain NaN/Inf")

    dataset = TensorDataset(torch.from_numpy(scaled))
    return dataset, labels


def evaluate_dataset(
    model: DFFForImageClassification,
    dataset: TensorDataset,
    labels: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    predictions: list[float] = []
    model.eval()
    with torch.no_grad():
        for (features,) in loader:
            features = features.to(device, non_blocking=True)
            outputs = model(topological_features=features)
            predictions.extend(
                outputs.logits.detach().float().cpu().view(-1).tolist()
            )

    pred = np.asarray(predictions, dtype=np.float64)
    true = labels["true_ba"].to_numpy(dtype=np.float64)
    if len(pred) != len(true):
        raise RuntimeError("Prediction count does not match label count.")

    pred_std = float(np.std(pred))
    if pred_std <= 1e-12:
        raise RuntimeError(
            "Predictions are effectively constant; PCC is not meaningful."
        )

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    pcc = float(pearsonr(true, pred).statistic)
    metrics = {
        "n": int(len(true)),
        "rmse_kcal_mol": rmse,
        "mae_kcal_mol": mae,
        "pcc": pcc,
        "prediction_std": pred_std,
    }
    predictions_frame = pd.DataFrame(
        {
            "sample_id": labels["sample_id"],
            "normalized_id": labels["normalized_id"],
            "true_ba": true,
            "predicted_ba": pred,
            "error": pred - true,
            "absolute_error": np.abs(pred - true),
            "squared_error": (pred - true) ** 2,
        }
    )
    return metrics, predictions_frame


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    formal_summary = validate_inputs(checkpoint)

    output_dir = (
        WORK_ROOT
        / "paper_outputs"
        / "03_aligned_mamba"
        / "casf_evaluation"
        / args.output_name
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}\n"
            "Independent CASF evaluation will not overwrite prior results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).resolve()
    script_snapshot = output_dir / "evaluation_script_snapshot.py"
    shutil.copy2(script_path, script_snapshot)

    command = " ".join(shlex.quote(item) for item in sys.argv)
    (output_dir / "command.txt").write_text(
        command + "\n",
        encoding="utf-8",
    )

    started = time.time()
    scaler = joblib.load(SCALER_PATH)
    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features not in (None, 6 * 100 * 143):
        raise ValueError(
            f"Scaler n_features_in_={scaler_features}, expected {6*100*143}"
        )

    model, config = build_model(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_metrics: dict[str, dict[str, float]] = {}
    for dataset_name, paths in DATASETS.items():
        print(f"\nEvaluating {dataset_name}...")
        dataset, labels = prepare_dataset(
            paths["features"],
            paths["labels"],
            scaler,
        )
        metrics, predictions = evaluate_dataset(
            model,
            dataset,
            labels,
            args.batch_size,
            device,
        )
        all_metrics[dataset_name] = metrics
        file_tag = dataset_name.lower().replace("-", "")
        predictions.to_csv(
            output_dir / f"{file_tag}_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(
            f"{dataset_name}: n={metrics['n']} "
            f"RMSE={metrics['rmse_kcal_mol']:.6f} "
            f"MAE={metrics['mae_kcal_mol']:.6f} "
            f"PCC={metrics['pcc']:.6f} "
            f"pred_std={metrics['prediction_std']:.6f}"
        )

    manifest = {
        "status": "completed",
        "scope": (
            "Independent external evaluation of the frozen ordered CLS-last "
            "Aligned-Mamba seed-0 checkpoint. No training, tuning, or "
            "checkpoint selection occurred."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "formal_training_summary": str(FORMAL_SUMMARY),
        "formal_training_status": formal_summary.get("status"),
        "formal_training_best_step": formal_summary.get("best_step"),
        "formal_internal_best_rmse": formal_summary.get(
            "best_validation_rmse"
        ),
        "evaluation_script_snapshot": str(script_snapshot),
        "evaluation_script_sha256": sha256_file(script_snapshot),
        "scaler": str(SCALER_PATH),
        "batch_size": args.batch_size,
        "device": str(device),
        "environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "model_config": {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "image_size": list(config.image_size),
            "patch_size": list(config.patch_size),
            "num_channels": config.num_channels,
            "pooler_type": config.pooler_type,
            "cls_token_position": getattr(
                config, "cls_token_position", "first"
            ),
            "d_state": getattr(config, "d_state", None),
            "d_conv": getattr(config, "d_conv", None),
            "expand": getattr(config, "expand", None),
        },
        "datasets": {
            name: {
                "features": str(paths["features"]),
                "labels": str(paths["labels"]),
            }
            for name, paths in DATASETS.items()
        },
        "metrics": all_metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "casf_metrics.json", all_metrics)
    write_json(output_dir / "evaluation_manifest.json", manifest)

    rows = [{"dataset": name, **metrics} for name, metrics in all_metrics.items()]
    pd.DataFrame(rows).to_csv(
        output_dir / "casf_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== Ordered CLS-last Mamba CASF evaluation summary ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\noutputs: {output_dir}")
    print("\nPASS: frozen ordered CLS-last Mamba seed-0 CASF evaluation completed.")
    print(
        "SCOPE: independent external benchmark results for one formal Mamba "
        "seed; not multi-seed or ensemble results."
    )


if __name__ == "__main__":
    main()
