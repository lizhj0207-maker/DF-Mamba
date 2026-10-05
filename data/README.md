# Datasets

DF-Mamba follows the protein-ligand datasets and Directed Flag topological representation used in the CAPTURE/DFFormer framework.

Large raw structure archives and locally generated processed feature tensors are **not redistributed** in this repository. Users should obtain the source data from the original providers and/or the CAPTURE data release, then prepare the required topological features with the supplied preprocessing code.

## Included label files

This repository includes the following small label files under `data/labels/`:

- `v2020_general_exclude_core_label.csv`
- `CASF2007_core_test_label.csv`
- `CASF2013_core_test_label.csv`
- `CASF2016_core_test_label.csv`
- `SARS_CoV2_CoV_labels.csv`

These files contain identifiers and corresponding binding-affinity labels used by the training/evaluation scripts. They are not processed model-feature tensors.

## Source datasets

### Combined PDBbind pretraining resource

The original CAPTURE release describes a combined PDBbind/CASF resource containing 19,513 unique protein-ligand complexes after duplicate removal. Binding-affinity labels are not used during self-supervised pretraining.

- PDBbind: http://www.pdbbind.org.cn/
- CAPTURE precomputed Directed Flag features:
  https://weilab.math.msu.edu/Downloads/CAPTURE/DFFeature_large.npy

### PDBbind v2020 fine-tuning set

The supervised fine-tuning set contains 18,904 complexes from the PDBbind v2020 general set after excluding the CASF-2007, CASF-2013, and CASF-2016 core sets.

- CAPTURE benchmark labels:
  https://weilab.math.msu.edu/Downloads/CAPTURE/Benchmarks_labels.zip

### CASF benchmark sets

External evaluation uses:

- CASF-2007 core set: 195 complexes
- CASF-2013 core set: 195 complexes
- CASF-2016 core set: 285 complexes

Labels are available through the CAPTURE benchmark-label release:

https://weilab.math.msu.edu/Downloads/CAPTURE/Benchmarks_labels.zip

Raw structures can be obtained from PDBbind.

### SARS-CoV / SARS-CoV-2 Mpro resource

The original CAPTURE repository reports a SARS-CoV2/CoV resource containing 203 inhibitor complexes with corresponding binding affinities:

https://weilab.math.msu.edu/Downloads/CAPTURE/SARS_CoV2_CoV_structures.zip

The local label table included in this repository contains the entries available in the experimental package used for the present study. Locally generated processed structural-feature tensors are not redistributed.

## Feature preparation

Protein-ligand structures are converted to Directed Flag Laplacian topological representations using the preprocessing implementation provided in:

`code_pkg/DF_embedding/DirectedFlagComplex_laplacian.py`

Processed `.npy` feature tensors generated locally are intentionally not included in this repository.

## Files intentionally not included

The repository does not include:

- locally generated `.npy` or `.npz` feature tensors
- model checkpoints
- prediction files
- PCC, RMSE, MAE, or other generated metric files
- training histories or logs
- temporary preprocessing outputs

## Original CAPTURE repository

https://github.com/WeilabMSU/CAPTURE

If you use CAPTURE data resources or the Directed Flag representation, please follow the citation requirements of the original CAPTURE project.
