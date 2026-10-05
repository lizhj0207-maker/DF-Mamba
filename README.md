# DF-Mamba

Official implementation of **DF-Mamba: Directed Flag Topology-Aligned Selective State-Space Learning for Protein-Ligand Binding Affinity Prediction**.

DF-Mamba combines Directed Flag topological representations of protein-ligand complexes with a Mamba-based selective state-space encoder for protein-ligand binding affinity prediction.

## Repository Structure

```text
DF-Mamba/
├── code_pkg/
│   ├── DF_embedding/
│   │   ├── __init__.py
│   │   └── DirectedFlagComplex_laplacian.py
│   └── DF_transformer/
│       ├── __init__.py
│       ├── configuration_dff.py
│       ├── modeling_dff_mamba_aligned.py
│       └── modeling_dff_mamba_ordered.py
├── scripts/
│   ├── training/
│   │   ├── run_aligned_mamba_pretraining.py
│   │   └── run_aligned_mamba_finetuning_cls_last.py
│   └── evaluation/
│       ├── run_mamba_cls_last_casf_seed0.py
│       └── run_mamba_mpro_5fold.py
└── data/
    ├── README.md
    └── labels/
        ├── CASF2007_core_test_label.csv
        ├── CASF2013_core_test_label.csv
        ├── CASF2016_core_test_label.csv
        ├── SARS_CoV2_CoV_labels.csv
        └── v2020_general_exclude_core_label.csv
```

## Model

DF-Mamba contains two main training stages.

### Self-Supervised Pretraining

- Directed Flag topological representations
- Mamba encoder
- 75% random masking
- masked-patch reconstruction

### Supervised Affinity Fine-Tuning

- complete ordered filtration sequence
- CLS token appended at the end of the sequence
- binding-affinity regression using the pretrained Mamba encoder

## Datasets

Small label files used by the training/evaluation scripts are included under `data/labels/`.

Large raw structures and locally processed feature tensors are not redistributed directly in this repository. Dataset sources and preparation information are provided in:

[data/README.md](data/README.md)

The source datasets are derived primarily from PDBbind, CASF benchmark sets, and resources released with the original CAPTURE/DFFormer project.

Original CAPTURE repository:

https://github.com/WeilabMSU/CAPTURE

## Usage

### Self-Supervised Pretraining

```bash
python scripts/training/run_aligned_mamba_pretraining.py
```

### Supervised Fine-Tuning

```bash
python scripts/training/run_aligned_mamba_finetuning_cls_last.py
```

### CASF Evaluation

```bash
python scripts/evaluation/run_mamba_cls_last_casf_seed0.py
```

### Mpro Five-Fold Evaluation

```bash
python scripts/evaluation/run_mamba_mpro_5fold.py
```

## Path Configuration

The current scripts preserve the local path configuration used during the original experiments.

Before running the code on another machine, users should modify the local data, checkpoint, and output paths according to their own environment.

## Data and Outputs

Locally generated experimental outputs are not included in this repository, including:

- model checkpoints
- prediction files
- training histories
- temporary preprocessing files
- locally generated feature tensors

Please refer to [data/README.md](data/README.md) for the source datasets and data-availability notes.

## Acknowledgements

This implementation builds upon the Directed Flag representation and code framework released with CAPTURE/DFFormer.

Original project:

https://github.com/WeilabMSU/CAPTURE
