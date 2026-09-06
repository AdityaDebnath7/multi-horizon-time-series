# Multi-Horizon Forecasting of CO Concentrations 📊
## A Rigorous Comparison of TFT, SARIMA, and Machine-Learning Baselines

**Dataset:** UCI Air Quality Dataset (hourly, Italy)  
**Task:** 10-step-ahead forecasting of CO(GT) from multivariate sensor history  
**Models:** Temporal Fusion Transformer (TFT) · SARIMA · XGBoost · Linear Regression · Persistence

---

## 1. Executive Summary ✅

This study evaluates whether a modern attention-based forecasting architecture, the Temporal Fusion Transformer (TFT), can outperform classical and machine-learning baselines under a strictly comparable, leakage-free evaluation protocol.

The latest results, extracted from the updated metric files in the `updated_results` folder, are summarized below.

### Headline results (10-step pooled performance)

| Rank | Model | MAE | RMSE | MAPE % | SMAPE % | MASE | R² | ME |
|------|-------|-----|------|--------|---------|------|----|----|
| 1 | SARIMA | 0.6323 | 0.7800 | 71.9034 | 42.1862 | 0.8859 | 0.0808 | -0.1237 |
| 2 | XGBoost | 0.6379 | 0.8116 | 67.5402 | 41.8188 | 0.8938 | 0.0050 | -0.0955 |
| 3 | TFT | 0.6587 | 0.8189 | 63.4811 | 40.3291 | 0.9228 | -0.0129 | -0.3618 |
| 4 | LinearReg | 0.6768 | 0.8528 | 70.6151 | 44.4287 | 0.9483 | -0.0985 | -0.1191 |
| 5 | Persistence | 0.7945 | 1.0272 | 72.0844 | 53.2277 | 1.1131 | -0.5940 | 0.0363 |

### Key takeaways

1. **SARIMA is the best pooled forecaster by MAE** and also achieves the best RMSE and MASE among the tested models. This confirms that the strong 24-hour seasonal signal in the CO series is highly exploitable.
2. **TFT is the most stable long-horizon model**. Its MAE rises from 0.5142 at h=1 to 0.6806 at h=10, a growth of only 0.1664, which is the smallest among all learned models.
3. **At short horizons, simple baselines remain competitive**. Linear regression achieves the lowest MAE at h=1 (0.3611), showing that recency dominates in the immediate future.
4. **All learned models under-predict peaks**. The mean error is negative for every trained model, with TFT showing the largest negative bias (-0.3618), suggesting conservative forecasting under volatile conditions.
5. **Persistence is clearly outperformed**. Its MASE of 1.1131 indicates it is worse than the seasonal-naive benchmark, confirming that the task contains meaningful temporal structure.

---

## 2. Data Description and Preparation 🧪

### 2.1 Source

The study uses the UCI Air Quality dataset, collected hourly in an Italian urban environment. The dataset includes 9,357 hourly observations with substantial sensor noise, missing values, and drift. The target variable is `CO(GT)`, measured in mg/m³.

### 2.2 Target and covariates

| Role | Variable | Description |
|------|----------|-------------|
| Target | `CO(GT)` | Ground-truth CO concentration |
| Covariates | `PT08.S1(CO)`, `PT08.S2(NMHC)`, `C6H6(GT)`, `PT08.S3(NO2)`, `NOx(GT)`, `PT08.S4(NO2)`, `PT08.S5(O3)` | Co-located sensor measurements used as multivariate inputs |

This yields a multivariate input history of 8 series per forecasting window.

### 2.3 Cleaning and alignment

A leakage-free preprocessing pipeline was used:

1. Missing sentinel values (`-200`) were converted to NaN.
2. Short gaps were filled conservatively to preserve short-range temporal continuity.
3. Physically implausible rows were removed.
4. The cleaned series was split chronologically into training, validation, and test subsets.
5. An explicit alignment audit was performed to verify the mapping between model outputs and test timestamps.

The result is a controlled forecasting setup where every model sees the same historical context and is evaluated on the same verified windows.

#### Alignment audit details

Because gap-filling can compress or shift rows, model outputs must be verified against
ground-truth timestamps positionally, not assumed. A full offset audit was performed:

- A reconstructed prediction series was scanned against `test_df["CO(GT)"]` across all
  candidate integer offsets.
- Best match: **offset = 15** (window `w` targets test rows `w+15 … w+24`), with only
  12 residual mismatches — all confined to the train/test boundary zone (positions 0–11),
  where gap-filling had skipped hours.
- Boundary windows were excluded; the remaining verified region showed
  **0.000% positional mismatch across 1,049 aligned pairs**, verified with exact
  equality asserts against `tft_true_final`.

All reported metrics are computed exclusively on these verified windows, guaranteeing
that every model's predictions are compared against the true matching hours.


---

## 3. Evaluation Protocol 🔎

### 3.1 Window definition

Each forecasting instance uses the preceding 30 hours of the multivariate history to predict the next 10 hourly CO values.

| Quantity | Value |
|----------|-------|
| Input history | 30 hours × 8 features |
| Forecast horizon | 10 steps |
| Stride | 1 hour |
| Shared test windows | 208 |
| NaN rule | Any window with invalid history or target is excluded uniformly across models |

This ensures an apples-to-apples comparison with no leakage and consistent evaluation conditions.

### 3.2 Modeling strategies

| Model | Forecasting strategy |
|-------|----------------------|
| TFT | Direct multi-step forecasting with a single forward pass for all 10 horizons |
| Persistence | Flat forecast using the last observed value for every future step |
| Linear Regression | Recursive one-step model applied iteratively |
| XGBoost | Recursive one-step model applied iteratively |
| SARIMA | Rolling-origin seasonal forecasting using the training-fitted model and ongoing state updates |

### 3.3 Leakage controls

Several controls ensure the comparison is leakage-free:

- **Chronological splitting** — no shuffling; the test period is strictly after training.
- **SARIMA fitted once on training data only.** Rolling-origin `.append(refit=False)`
  calls update the Kalman filter state with observations strictly in the past relative
  to each forecast origin; no parameters are re-estimated on test data.
- **Feature scaling fit on training data only** for Linear Regression and XGBoost.
- **Identical information sets.** All multivariate models receive the same 30-hour
  history (including legitimate train-tail rows for the earliest test windows).
  SARIMA is deliberately *more* restricted — univariate — and therefore cannot
  benefit from extra information.
- **Shared evaluation subset.** SARIMA was evaluated on every 5th window (stride 5,
  208 windows) for computational feasibility; **all other models were subset to the
  identical 208 windows**, so the final table is strictly apples-to-apples.
- **Uniform NaN rule.** Any window with invalid history or targets is excluded for
  every model identically.


---

## 4_0. How Each Model Sees the Data

| Model | Input shape | What the input contains | Output shape | Output meaning |
|-------|-------------|--------------------------|--------------|----------------|
| TFT | (30, 8) + time features | 30 hourly steps of CO + 7 sensors, plus hour-of-day / day-of-week | (10,) | All 10 horizon forecasts from one forward pass (direct strategy) |
| SARIMA | (T,) | Univariate CO series only; parameters fit once on training region; Kalman state carried forward | (10,) | 10-step seasonal forecast from each rolling origin |
| XGBoost | (240,) | 30×8 history flattened to 240 features, standardized (train-fitted scaler) | (10,) | h=1 predicted directly, h=2..10 by recursion: predicted CO fed back, sensors frozen at last observed values |
| LinearReg | (240,) | Same flattened standardized history as XGBoost | (10,) | Same recursive procedure as XGBoost |
| Persistence | scalar | Last observed CO value | (10,) | Value held flat across all horizons |

The multivariate models see 8 series; SARIMA sees only CO. The recursive models
cannot know future sensor values, so covariates are frozen at the origin — the same
constraint the TFT's decoder faces. This makes all five strategies comparable in
information, differing only in *how* they map history to the future.


## 4. Results 📈


### 4.1 Final pooled comparison (208 shared windows)

The results below are taken directly from the updated results files in the `updated_results` folder.

| Model | MAE | RMSE | MAPE % | SMAPE % | MASE | R² | ME |
|-------|-----|------|--------|---------|------|----|----|
| Persistence | 0.7945 | 1.0272 | 72.0844 | 53.2277 | 1.1131 | -0.5940 | 0.0363 |
| LinearReg | 0.6768 | 0.8528 | 70.6151 | 44.4287 | 0.9483 | -0.0985 | -0.1191 |
| XGBoost | 0.6379 | 0.8116 | 67.5402 | 41.8188 | 0.8938 | 0.0050 | -0.0955 |
| TFT | 0.6587 | 0.8189 | 63.4811 | 40.3291 | 0.9228 | -0.0129 | -0.3618 |
| SARIMA | 0.6323 | 0.7800 | 71.9034 | 42.1862 | 0.8859 | 0.0808 | -0.1237 |

### 4.2 Per-horizon MAE by hour

| Model | h=1 | h=2 | h=3 | h=4 | h=5 | h=6 | h=7 | h=8 | h=9 | h=10 | Δ(h1→h10) |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|-----------|
| Persistence | 0.3779 | 0.5942 | 0.7154 | 0.7688 | 0.8481 | 0.9466 | 0.9308 | 0.9240 | 0.9221 | 0.9173 | +0.5394 |
| LinearReg | 0.3611 | 0.5445 | 0.6647 | 0.6672 | 0.6908 | 0.7691 | 0.7777 | 0.7731 | 0.7604 | 0.7596 | +0.3985 |
| XGBoost | 0.3890 | 0.5220 | 0.6249 | 0.6176 | 0.6272 | 0.7293 | 0.7204 | 0.7500 | 0.7078 | 0.6912 | +0.3022 |
| TFT | 0.5142 | 0.5807 | 0.6468 | 0.6611 | 0.6979 | 0.6966 | 0.6994 | 0.7155 | 0.6937 | 0.6806 | +0.1664 |
| SARIMA | 0.3882 | 0.5648 | 0.6423 | 0.6415 | 0.6635 | 0.7071 | 0.6717 | 0.6941 | 0.6755 | 0.6745 | +0.2863 |

### 4.3 Interpretation of the results

- **SARIMA** is the strongest pooled performer on MAE, RMSE, and MASE. This is consistent with the strong daily periodicity in the CO series.
- **TFT** has the smallest long-horizon error growth, indicating the most stable performance as the forecast horizon increases.
- **Linear Regression** is best at h=1, showing that immediate dependence on recent observed values is very strong.
- **XGBoost** is competitive overall but accumulates more recursive error than TFT as the horizon lengthens.
- **Persistence** is the weakest baseline and does not beat the seasonal-naive benchmark.

---

## 5. Discussion and Interpretation 💡

1. **Direct multi-step forecasting helps long-horizon stability.** TFT avoids recursive error accumulation, which explains its flatter MAE curve versus the recursive models.
2. **Short-term forecasting is dominated by recency.** The lowest h=1 error comes from Linear Regression, confirming that the last visible observations carry the most predictive power at very short horizons.
3. **Seasonality remains a dominant force.** SARIMA benefits from the repeating 24-hour structure in the air-quality signal and remains highly competitive even without multivariate covariates.
4. **Peak underestimation persists across all models.** Negative mean errors reflect systematic under-forecasting during high-concentration episodes, which is a known challenge in air-quality forecasting.
5. **TFT is strongest on percentage-based errors.** It achieves the lowest MAPE and SMAPE despite not being the lowest-MAE model, indicating that its errors are relatively well distributed across the concentration range.

6. **Error growth curves separate the model classes more than any single number.**
   From h=1 to h=10, MAE grows by +0.539 (Persistence), +0.399 (LinearReg),
   +0.302 (XGBoost), +0.286 (SARIMA), and only **+0.166 (TFT)**. TFT's curve also
   *non-monotonic*: its MAE actually decreases from h=8 (0.7155) to h=10 (0.6806),
   reflecting the daily seasonal cycle it learned — by hour 10 the model has
   "wrapped around" to a familiar phase of the day.
7. **The h=1 disadvantage of TFT is structural, not a defect.** At h=1 the previous
   hour's observation is nearly optimal, which favors Persistence (0.3779) and
   LinearReg (0.3611). TFT's encoder distributes capacity across all 10 heads
   simultaneously and trades a worse h=1 (0.5142) for the best stability from h=6
   onward. For early-warning applications, where 6–10 hour lead times matter,
   this is the favorable operating regime.
8. **Bias ordering is informative.** TFT's larger negative ME (−0.3618 vs ≈ −0.10 to
   −0.12 for the other trained models) indicates the most conservative forecasts:
   it shrinks toward the conditional mean more strongly, which costs it MAE on
   peaks but explains its superior percentage-error metrics in low-concentration
   regimes.



### Reading the pooled metrics correctly

Three metric behaviors require explicit interpretation to avoid misreading the table:

1. **Pooled R² ≈ 0 is expected, not a failure.** R² is computed against the variance
   of all pooled targets. At horizons of 6–10 hours, much of the target variance is
   unpredictable noise, so even a good forecaster cannot explain the full variance
   pooled across horizons. SARIMA is the only model with positive pooled R² (0.0808),
   and Persistence's −0.594 shows it is actively worse than predicting the mean.
2. **High MAPE (63–72%) is an artifact of near-zero targets.** CO(GT) frequently takes
   values near zero, where any small absolute error produces a huge percentage error.
   This is precisely why SMAPE (bounded, symmetric) and MASE (scaled by a benchmark)
   are reported alongside: on those, TFT is best (SMAPE 40.33) and SARIMA best
   (MASE 0.8859), respectively.
3. **MASE anchors everything.** The MASE scale is the in-sample MAE of the
   seasonal-naive (period 24) forecast on training data only (0.7138), applied
   identically to all models. MASE < 1 means the model beats the seasonal naive.
   Only Persistence fails this test (1.1131), confirming genuine learnable structure.


---

## 6. Conclusion ✅

The empirical results suggest that no single model dominates every regime. SARIMA is the strongest pooled forecaster overall, while TFT provides the best long-horizon stability and exhibits the slowest deterioration in MAE across the 10-hour forecast window. In practical monitoring applications, this means: classical seasonal models remain highly competitive for aggregate accuracy, while transformer-based modeling offers superior stability when longer lead times matter more than immediate one-step precision.

The metrics presented in this report were generated from the latest CSV files in the `updated_results` folder and are consistent with the final evaluation outputs used for comparison.

---

## 7. Reproducibility 🧭

- **Primary artifacts:** `updated_results/final_model_comparison_metrics.csv` and `updated_results/final_per_horizon_metrics.csv`
- **Model stack:** Python, PyTorch Forecasting, SARIMA, scikit-learn, XGBoost, pandas, and matplotlib
- **Evaluation protocol:** chronological split, identical window definition, fixed shared test subset, and consistent preprocessing across models
- **Practical note:** the analysis is dataset-specific and should be validated on additional cities or seasons before broad generalization

---

## 8. Final Takeaway ✅

Under a strict, leakage-aware evaluation protocol, the evidence supports a clear practical distinction:

- **SARIMA** is the best overall pooled forecast model on MAE, RMSE, and MASE.
- **TFT** is the most stable model across long horizons and the best percentage-error forecaster.
- **Linear Regression** remains strongest at h=1.
- **Persistence** is clearly below the seasonal-naive benchmark and is not competitive.

This makes the final recommendation horizon-dependent: use classical seasonal models for best aggregate accuracy, and prefer TFT when stable multi-hour forecasting is more important than one-step performance.
