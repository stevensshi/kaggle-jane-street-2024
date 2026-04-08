# Phase 6: Online Learning — Findings

## Setup

- **Base model**: GRU lag-residual (Phase 5, Val R² = 0.883)
- **Val data**: dates 1189-1443 (9.4M rows, 246,840 time steps, 255 days)
- **Simulation**: Step-by-step GRU inference mimicking competition loop
- **Per-symbol hidden states**: maintained and reset at day boundaries

---

## Results

| Experiment                     | Val R²     | Δ vs Frozen | Notes                    |
|-------------------------------|------------|-------------|--------------------------|
| **Frozen baseline**           | **0.882852** | —         | Step-by-step matches batch |
| All-layers SGD lr=1e-6        | 0.882815   | −0.000037   | Negligible change         |
| Head-only SGD lr=1e-5         | 0.882604   | −0.000248   | Slight degradation        |
| All-layers SGD lr=1e-5        | 0.882617   | −0.000235   | Slight degradation        |
| All-layers SGD lr=1e-4        | 0.882019   | −0.000833   | Worse                     |
| Head-only SGD lr=1e-4         | 0.881967   | −0.000885   | Worse                     |
| Head-only SGD lr=1e-3         | 0.879908   | −0.002944   | Much worse                |
| Lag EMA α=0.01                | 0.857672   | −0.025180   | Destructive               |
| Lag EMA α=0.05                | 0.855536   | −0.027316   | Destructive               |
| Lag EMA α=0.10                | 0.850157   | −0.032695   | Destructive               |

---

## Key Findings

### 1. Online learning does NOT improve on this validation set

Every online adaptation scheme performed worse than the frozen baseline. The
gradient-based updates caused small degradation (−0.0002 to −0.003) while the
lag EMA tracking was destructive (−0.025 to −0.033).

### 2. The frozen model is already well-adapted

The validation data follows the same distribution as training data (lag autocorrelation
≈0.90 persists). There is no distribution shift for online learning to adapt to.

### 3. Lag EMA is particularly harmful

Dynamically adjusting the lag_scale via EMA of actual/predicted ratios injects noise
and destabilizes the model. The learned lag_scale from training is more reliable than
any moving estimate.

### 4. Step-by-step inference matches batch inference

Frozen baseline R² = 0.882852 vs batch R² = 0.882851. This confirms the step-by-step
GRU implementation (maintaining per-symbol hidden states, resetting at day boundaries)
is correct.

### 5. Smaller learning rates are less harmful

The degradation is monotonic with learning rate: lr=1e-6 barely hurts, lr=1e-3 hurts
significantly. This pattern suggests the updates add noise rather than useful adaptation.

---

## Implications

**For this competition's historical data**: Skip online learning. The frozen GRU is optimal.

**For real future data** (where competition best score ≈ 0.013): Online learning could
theoretically help if the lag autocorrelation drops or regime changes occur. But our
validation set can't test this — the lag correlation of 0.90 is stable across all dates
in our data.

The low competition score (0.013 vs our 0.88) suggests the lag signal completely
disappears on truly future data. In that scenario, the entire model architecture
(which relies on lag1_r6 as the dominant signal) would need rethinking, not just
online adaptation.

---

## Files

- `reports/online_learning/results.json` — numeric results
- `src/online_learning.py` — full online learning simulation
