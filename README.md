# Physics-Informed Neural State Estimation from fNIRS

Deep learning system for recovering cardiovascular state trajectories from a single short-channel fNIRS signal using a learned forward observation model, causal WaveNet inverse encoder, and innovation-feedback drift correction.

This project demonstrates applied AI/Data Science skills in probabilistic modelling, time-series forecasting, biomedical signal processing, autoregressive inference, and closed-loop neural state estimation.

## Key Results

| Result | Value |
|---|---:|
| Forward model R² | 0.719 |
| Teacher-forced student R² | ~0.97 |
| Closed-loop rollout R² | 0.637 |
| Best recovered signal | Systolic BP, R² = 0.773 |
| Short-horizon closed-loop R² | 0.83 over ~8 seconds |
| Error growth | Bounded / non-divergent |

## What I Built

I built a two-stage neural system that estimates four cardiovascular variables from one optical fNIRS signal:

- Diastolic blood pressure
- Systolic blood pressure
- Cardiac output
- Stroke volume

The model uses a physics-informed observation-consistency loop:

1. A conditional normalizing-flow teacher learns the forward relationship: cardiovascular state → optical signal.
2. A causal WaveNet student learns the inverse relationship: optical signal → cardiovascular state.
3. During rollout, the student’s predicted state is passed back through the frozen teacher.
4. The difference between the teacher-predicted optical signal and the observed optical signal becomes an innovation signal for drift correction.

```text
Short-channel fNIRS → causal features → WaveNet student → cardiovascular state estimate
                                      ↓
                           frozen forward teacher
                                      ↓
                     observation mismatch / innovation feedback

## How It Works

The teacher and student solve **opposite problems**:

- **Teacher** (Stage 1): Given cardiovascular state x → predict optical signal y. Trained with both available.
- **Student** (Stage 2): Given optical signal y → recover cardiovascular state x. At inference, only y is available.

The student never imitates the teacher directly. Instead, the student's predicted states are passed through the frozen teacher to check if they produce the correct optical signal. If the prediction is wrong, the teacher's output will disagree with what was observed — and that disagreement is the learning signal.

```
Student predicts x̂  →  Frozen teacher maps x̂ → ŷ  →  Compare ŷ vs y_obs  →  Correct student
```

At inference, this same mechanism provides real-time drift correction: the teacher detects when predictions have gone wrong by comparing expected vs observed optical signals.

---

## Project Structure

```
├── core_flow.py              # Normalizing flow architecture (teacher internals)
├── exp_base_static.py        # Data loading, feature engineering, teacher model definition
├── prior_student.py          # Student architecture definition (used by both training and inference)
├── Encoder.py                # Training pipeline (runs both Stage 1 and Stage 2)
├── Inference.py              # Standalone inference (closed-loop rollout on unseen data)
├── README.md
│
└── output/                   # Data files + training outputs
    ├── ShortChannel.csv      # Input: fNIRS optical signal (~11,000 samples, 1 subject)
    ├── DiastolicBP.csv       # Ground truth: diastolic blood pressure
    ├── SystolicBP.csv        # Ground truth: systolic blood pressure
    ├── CardiacOutput.csv     # Ground truth: cardiac output
    ├── StrokeVolume.csv      # Ground truth: stroke volume
    │
    ├── ygx_ckpt.pt           # Saved teacher model weights (Stage 1)
    ├── scaler.json           # Standardisation statistics (means, std devs for all channels)
    ├── x_norm_stats.json     # Raw normalisation stats for the 4 cardiovascular channels
    ├── y_memory_encoder.pt   # Saved student model weights + full training config (Stage 2)
    └── decoder_pack.pt       # Self-contained export (teacher + student + all scalers)
```

---

## What Each File Does

### Source Code

| File | Role |
|---|---|
| `core_flow.py` | Defines the **normalizing flow architecture** used inside the teacher. Contains the mathematical building blocks: coupling layers, spline transforms, activation normalisation, and the change-of-variables likelihood computation. This file is purely architectural — it has no knowledge of fNIRS or cardiovascular channels. |
| `exp_base_static.py` | Handles **data loading and feature engineering**. Reads the 5 CSV files, joins them on timestamps, builds the 75 causal observation features from ShortChannel (lags, derivatives, EMAs, bandpass energies, cardiac phase), standardises the 4 cardiovascular channels, and defines the teacher model class (`YGivenXModel`) that wraps the normalizing flow. |
| `prior_student.py` | Defines the **student encoder architecture** — the WaveNet backbone, the y-only observation backbone, all drift correction mechanisms (bias integrator, innovation feedback, kappa mixing, gate). This is the single source of truth: both `Encoder.py` and `Inference.py` use this file to build the student, ensuring the architecture matches exactly when loading saved weights. |
| `Encoder.py` | The **training script**. Runs the full pipeline: trains the teacher (18 epochs), freezes it, then trains the student (30 epochs) with a structured curriculum that progresses from observation-only pretraining to fully closed-loop rollout. Also exports all saved artefacts and runs Stage 1 evaluation. |
| `Inference.py` | The **standalone inference script**. Loads the trained student and optionally the frozen teacher, then runs a closed-loop rollout on data using only the ShortChannel signal. Produces predictions in physical units, computes evaluation metrics, and generates diagnostic plots. |

### Data Files (in `output/`)

All CSV files contain ~11,000 samples from a single subject recorded at approximately 6 Hz during a public speaking paradigm. Each CSV has two columns: `Time` and the measurement value.

| File | What it contains |
|---|---|
| `ShortChannel.csv` | The fNIRS optical signal — **this is the only input** the student sees at inference. Everything else is derived from this single signal. |
| `DiastolicBP.csv` | Diastolic blood pressure recordings — used as ground truth during training and for evaluation metrics. |
| `SystolicBP.csv` | Systolic blood pressure recordings — same role as above. |
| `CardiacOutput.csv` | Cardiac output recordings — same role as above. |
| `StrokeVolume.csv` | Stroke volume recordings — same role as above. |

### Saved Weights and Metadata (in `output/`)

These are generated by `Encoder.py` during training and consumed by `Inference.py` at inference time.

| File | What it contains |
|---|---|
| `ygx_ckpt.pt` | **Teacher model weights.** The frozen Stage 1 normalizing flow that learned P(y\|x). Used during student training for observation anchoring and at inference for innovation feedback. |
| `scaler.json` | **Standardisation statistics.** Mean and standard deviation for every channel (cardiovascular inputs, optical signal, engineered features). Required to transform raw data into the standardised space the models expect, and to convert predictions back to physical units. |
| `x_norm_stats.json` | **Raw normalisation stats** for the 4 cardiovascular channels specifically. Subset of what `scaler.json` contains, saved separately for convenience. |
| `y_memory_encoder.pt` | **Student model payload.** Contains the trained student weights, full architecture configuration (layer sizes, number of blocks, gate floor value), feature scaler statistics, and all training hyperparameters. Everything needed to reconstruct the student exactly as it was at the end of training. |
| `decoder_pack.pt` | **Complete self-contained export.** Bundles the student weights, teacher weights, all scalers, and precomputed rollout streams into a single file. Designed so a downstream decoder can be trained or inference can run without needing any other files. |

---

## How to Run

### Training

Run `Encoder.py`. It executes the full pipeline automatically:

1. Trains the teacher on the forward mapping (18 epochs)
2. Freezes the teacher
3. Trains the student on the inverse mapping (30 epochs)
4. Exports all saved artefacts to `output/`

### Inference

Run `Inference.py` after training. Configure these paths at the bottom of the file:

```python
FOLDER          = "./output"                          # Folder with CSV data files
ENCODER_PATH    = "./output/y_memory_encoder.pt"      # Student weights
SCALER_PATH     = "./output/scaler.json"              # Standardisation stats
YGX_CKPT_PATH   = "./output/ygx_ckpt.pt"             # Teacher weights (only if innovation ON)
USE_INNOVATION  = True                                 # Use teacher for drift correction?
```

### Requirements

- **GPU**: CUDA-enabled GPU recommended
- **Dependencies**: PyTorch, NumPy, SciPy, Pandas, Matplotlib, scikit-learn

---

## Performance

| Metric | Value |
|---|---|
| Teacher R² (forward mapping) | 0.719 |
| Student R² (teacher-forced) | ~0.97 |
| Student R² (closed-loop rollout, training data) | 0.637 |
| Best single dimension R² | 0.773 (Systolic BP) |
| Short-horizon R² (8 seconds) | 0.83 |
| Error growth | Bounded, non-divergent |

Degradation on unseen data (R² = −0.56) is attributed to the forward mapping shifting during the public speaking task — the relationship between cardiovascular state and optical signal changes as the subject's autonomic state evolves.

---
