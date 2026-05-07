# Code for: [Paper Title]

*NeurIPS 2026 Submission — Anonymous*

This repository contains the code accompanying the paper. It is provided for
reproducibility. The data used in this work is from a private dataset
and **cannot be redistributed**; code is provided as-is for inspection and
adaptation.

---

## Overview

We propose a few-shot EMG gesture recognition system combining **Model-Agnostic
Meta-Learning (MAML++)** with a **Mixture-of-Experts (MoE) CNN encoder**. The
system adapts to new users from a single labeled trial per gesture class
(1-shot, 3-way). The MoE pool consists of 22 CNN experts with soft top-9
routing, trained end-to-end with episodic meta-learning. Inputs are
multimodal: 16-channel surface EMG + 72-channel IMU, windowed at 64 time steps.

The primary model (**M0**) and all ablations are in the root directory.
`ablation_config.py` is the single source of truth for all hyperparameters;
individual scripts only override what their ablation changes.

---

## Repository Structure

```
.
├── ablation_config.py          # Shared hyperparameters + model builders + eval utilities
├── M0_full_model.py            # Primary model: MAML++ + MoE  [main result]
├── A2_no_maml_no_moe.py        # Ablation: supervised pretraining, single encoder (param-matched to all M0 experts)
├── mamlpp.py                   # MAML++ training loop (inner/outer, LSLR, MSL)
├── MOE_encoder.py              # MoE CNN encoder (encoder-placement and middle-placement variants)
├── system/
│   ├── MAML/
│   │   ├── maml_data_pipeline.py   # Episodic dataloader (MetaGestureDataset)
│   │   └── mamlpp.py               # adapt-and-eval at test time
│   ├── MOE/
│   │   └── MOE_encoder.py
│   ├── pretraining/
│   │   ├── pretrain_data_pipeline.py
│   │   ├── pretrain_trainer.py
│   │   ├── pretrain_models.py
│   │   └── pretrain_finetune.py
│   └── fixed_user_splits/
│       └── hpo_strat_kapanji_split.json
└── dataset/
    └── meta-learning-sup-que-ds/   # Preprocessed tensor dicts (not redistributed)
```

---

## Ablation Models

| ID | MAML | MoE | Notes |
|----|------|-----|-------|
| **M0** | ✓ | ✓ | Full model (EncoderMoE) — primary result |
| A1 | ✗ | ✓ | Supervised pretraining + MoE |
| A2 | ✗ | ✗ | Supervised pretraining, single encoder (param-matched to *all* M0 experts combined) |
| A4 | ✓ | ✗ | MAML, single encoder (param-matched to *all* M0 experts combined) |

A2 vs A4 isolates the contribution of MAML at equal model capacity.  
A4 vs M0 isolates the contribution of MoE at equal total expert capacity.

---

## Evaluation Protocol

All final results use **Leave-2-Subjects-Out (L2SO)**. For fold *i*:
- test subject = `all_PIDs[i]`
- val subject = `all_PIDs[(i+1) % N]`
- train subjects = everyone else

The fixed `hpo_test_split` (24/4/4) was used only for HPO (Optuna, Trial 89)
and should not be used for ablation comparisons — doing so would conflate HPO
effects with ablation effects.

Test evaluation uses 500 episodic episodes (1-shot, 3-way) per subject.
Reported accuracy is mean ± std across subjects.

---

## Hyperparameters

All hyperparameters are from **Optuna Trial 89** (`val_acc = 90.05%`). Key values:

| Parameter | Value |
|-----------|-------|
| `outer_lr` | 1.951e-4 |
| `weight_decay` | 8.874e-4 |
| `maml_inner_steps` | 10 |
| `maml_alpha_init_eval` | 5.066e-3 |
| `num_experts` | 22 |
| `MOE_top_k` | 9 |
| `MOE_gate_temperature` | 1.529 |
| `cnn_base_filters` | 64 |
| `lstm_hidden` | 64 |
| `lstm_layers` | 3 (bidirectional) |

See `ablation_config.py` for the full set with pre-Trial-89 values annotated.

---

## Running the Code

> **Note:** The preprocessed dataset (`segfilt_rts_tensor_dict.pkl`) is not
> included. The code is provided for inspection and cannot be run without access
> to the original data.

**Environment variables expected:**
```bash
export CODE_DIR=/path/to/this/repo
export DATA_DIR=/path/to/data
export RUN_DIR=/path/to/output/dir
```

**Run the full model (all L2SO folds sequentially):**
```bash
python M0_full_model.py --test-procedure L2SO
```

**Run a single fold (e.g., for SLURM parallelism):**
```bash
python M0_full_model.py --test-procedure L2SO --fold-idx 0
```

**Run an ablation:**
```bash
python A2_no_maml_no_moe.py --test-procedure L2SO
```

---

## Dependencies

- Python 3.9+
- PyTorch ≥ 2.0
- NumPy, tqdm
- (HPO only) Optuna
