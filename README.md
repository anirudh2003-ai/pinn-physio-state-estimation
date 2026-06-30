# Physics-Informed Neural State Estimation from fNIRS

Deep learning system for recovering cardiovascular state trajectories from a single short-channel fNIRS signal using physics-informed time-series feature extraction, a learned forward observation model, a causal WaveNet inverse encoder, and innovation-feedback drift correction.

This project demonstrates applied AI/Data Science skills in probabilistic modelling, physiological signal processing, time-series forecasting, autoregressive inference, and closed-loop neural state estimation.
## Key Results

| Result | Value |
|---|---:|
| Forward observation model R² | 0.719 |
| Teacher-forced student R² | ~0.97 |
| Closed-loop rollout R² | 0.637 |
| Best recovered signal | Systolic BP, R² = 0.773 |
| Short-horizon closed-loop R² | 0.83 over ~8 seconds |
| Error growth | Bounded / non-divergent |

---

## Project Summary

This project tackles an ill-posed inverse problem: recovering four hidden cardiovascular variables from one noisy optical fNIRS signal.

The model estimates:

- Diastolic blood pressure
- Systolic blood pressure
- Cardiac output
- Stroke volume

Unlike a standard supervised regression model, this system uses a physics-informed observation-consistency loop. A forward model learns how cardiovascular state produces the observed optical signal, while an inverse model learns to recover the cardiovascular state from the optical signal alone.

```text
Short-channel fNIRS
        ↓
75 causal signal features
        ↓
Causal WaveNet student
        ↓
Cardiovascular state estimate
        ↓
Frozen forward teacher
        ↓
Observation mismatch / innovation feedback
        ↓
Drift-corrected closed-loop rollout
```

---

## Why This Matters

Short-channel fNIRS signals are often treated as nuisance physiological noise. This project shows that the same signal can contain recoverable cardiovascular information.

The main challenge is not just prediction accuracy, but maintaining stability when the model runs autoregressively, where each prediction becomes part of the next input. The system uses innovation feedback, an EMA bias integrator, kappa mixing, and bounded state updates to reduce long-horizon drift.

---

## Technical Highlights

- Physics-informed feature extraction from fNIRS time series
- 75 strictly causal features capturing cardiac, respiratory, trend, derivative, phase, EMA, rolling-statistic, and Kalman-smoothed dynamics
- Conditional normalizing-flow teacher for learning the forward observation density `P(y|x)`
- Causal WaveNet student for inverse cardiovascular state estimation
- Learned observation-consistency check using a frozen forward model
- Innovation-feedback correction inspired by Kalman filtering
- Gated EMA bias integrator for long-horizon drift correction
- Closed-loop autoregressive inference with no ground-truth physiology at test time

---

## Model Architecture

The system has two stages:

### Stage 1: Forward Teacher

The teacher learns:

```text
cardiovascular state x → optical fNIRS signal y
```

It uses a conditional normalizing flow with a Student-t residual head to model the noisy forward relationship between cardiovascular variables and the observed short-channel fNIRS signal.

### Stage 2: Inverse Student

The student learns:

```text
optical fNIRS signal y → cardiovascular state x
```

At inference, the student only receives the fNIRS signal. Its predicted cardiovascular state is passed through the frozen teacher. The mismatch between the teacher-predicted optical signal and the real optical signal becomes an innovation signal used for online drift correction.

```text
Student predicts x̂ → Frozen teacher predicts ŷ → Compare ŷ with y_obs → Correct student
```

---

## Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib

---

## Repository Structure

```text
├── core_flow.py              # Normalizing flow architecture for the teacher
├── exp_base_static.py        # Data loading, feature engineering, teacher definition
├── prior_student.py          # Student architecture: WaveNet, innovation feedback, bias correction
├── Encoder.py                # Full training pipeline
├── Inference.py              # Closed-loop inference script
├── README.md
│
└── output/
    ├── ShortChannel.csv      # fNIRS optical input signal
    ├── DiastolicBP.csv       # Ground-truth diastolic blood pressure
    ├── SystolicBP.csv        # Ground-truth systolic blood pressure
    ├── CardiacOutput.csv     # Ground-truth cardiac output
    ├── StrokeVolume.csv      # Ground-truth stroke volume
    │
    ├── ygx_ckpt.pt           # Teacher model weights
    ├── y_memory_encoder.pt   # Student model weights
    ├── decoder_pack.pt       # Self-contained inference export
    ├── scaler.json           # Standardisation statistics
    └── x_norm_stats.json     # Cardiovascular-channel normalisation stats
```

---

## How to Run

### 1. Install dependencies

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib
```

### 2. Train the model

```bash
python Encoder.py
```

This runs the full training pipeline:

1. Train the forward teacher model
2. Freeze the teacher
3. Train the inverse student model
4. Export model weights and scalers to `output/`

### 3. Run inference

```bash
python Inference.py
```

Configure these paths inside `Inference.py`:

```python
FOLDER          = "./output"
ENCODER_PATH    = "./output/y_memory_encoder.pt"
SCALER_PATH     = "./output/scaler.json"
YGX_CKPT_PATH   = "./output/ygx_ckpt.pt"
USE_INNOVATION  = True
```

---

## Performance

| Metric | Value |
|---|---:|
| Teacher R², forward mapping | 0.719 |
| Student R², teacher-forced | ~0.97 |
| Student R², closed-loop rollout | 0.637 |
| Best single output | Systolic BP, R² = 0.773 |
| Short-horizon rollout | R² = 0.83 over ~8 seconds |
| Error behaviour | Bounded, non-divergent |

The model performs strongly on short-horizon closed-loop inference and remains bounded over longer rollouts. Long-horizon degradation on unseen data is attributed to non-stationarity in the public-speaking task, where the physiological relationship between cardiovascular state and optical fNIRS signal changes over time.

---

## Limitations and Future Work

This is a research prototype evaluated on a single-subject recording. Cross-subject generalisation remains untested.

Future improvements:

- Multi-subject evaluation
- Subject-adaptive fine-tuning
- Online adaptation of the forward observation model
- Longer rollout training horizons
- More robust handling of task-induced non-stationarity
- Deployment as a lightweight inference package

---

## Relevance to AI/Data Science Roles

This project demonstrates experience with:

- Building end-to-end deep learning systems
- Designing architectures for noisy time-series data
- Solving inverse problems with probabilistic models
- Handling autoregressive drift and error accumulation
- Engineering causal features for physiological signal processing
- Evaluating models beyond standard train/test accuracy
- Communicating research results through reproducible code
