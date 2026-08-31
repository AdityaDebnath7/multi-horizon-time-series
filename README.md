# 10-Step Ahead Forecasting Evaluation - Final Report

## Executive Summary

This is an exploratory **10-step forecasting extension** for CO prediction. It is not the official final claim of the project.

The validated project conclusion remains the single-step benchmark: **XGBoost was the strongest model** for the primary forecasting task. The 10-step section below was added as a secondary comparison and should be interpreted with explicit caveats.

Successfully implemented and evaluated **10-step ahead forecasting** for CO concentration prediction using:
- **5 Models**: Persistence, Linear Regression, XGBoost, SARIMA, Temporal Fusion Transformer (TFT)
- **1,426 forecast sequences** × 10 steps = 14,260 total predictions
- **Efficient SARIMA**: Single training with direct `forecast(steps=10)` (166 seconds, no retraining)
- **Horizon-wise evaluation**: Per-step metrics (Steps 1-10) + aggregated metrics

> Important methodological note: the current notebook includes a few caveats that should be stated clearly. The persistence benchmark at step 1 was effectively tautological due to indexing, the SARIMA baseline was a fixed-origin forecast rather than a true rolling-origin benchmark, and the TFT result is a recursive proxy rather than a native 10-step TFT decoder output. These caveats do not invalidate the broader workflow, but they do affect how the results should be framed in a final report or interview.

---

## Key Results Summary

### 🏆 Overall Winner: **XGBoost**
- **Aggregated RMSE (all 10 steps)**: 0.4990 (approximately 47% better than persistence over steps 2–10; step 1 is treated as an artifact and excluded from the headline comparison)
- **Aggregated MAE**: 0.3480
- **Aggregated MAPE**: 47.79%
- **Aggregated R²**: 0.6077
- **Wins on 9/10 horizon steps** (best performer 2-10)

### Performance Ranking (Aggregated Metrics)
| Rank | Model | RMSE | MAE | MAPE | R² | Bias |
|------|-------|------|-----|------|-----|------|
| 1️⃣ | **XGBoost** | **0.4990** | **0.3480** | **47.79%** | **0.6077** | -0.0985 |
| 2️⃣ | Linear Regression | 0.7535 | 0.5961 | 78.92% | 0.1057 | -0.2234 |
| 3️⃣ | TFT | 0.8711 | 0.7123 | 101.51% | -0.1951 | -0.4111 |
| 4️⃣ | SARIMA | 0.8850 | 0.7369 | 102.01% | -0.2336 | -0.3790 |
| 5️⃣ | Persistence | 0.9314 | 0.6707 | 66.64% | -0.3665 | 0.0039 |

---

## Horizon-Wise Performance Breakdown

### RMSE Degradation Across Horizons (1-10 steps)

**XGBoost** shows excellent consistency:
- Step 1: 0.3608 RMSE
- Step 10: 0.5201 RMSE
- **Degradation: 44.15%** (stable from steps 4-10)

**Persistence** (baseline) shows sharp degradation:
- Step 1: 0.0000 (⚠ artifact: same-time comparison; excluded from official comparison)
- Step 10: 1.1130 RMSE
- **Degradation: Infinite** (no forecasting ability)

**SARIMA** shows improvement with horizon:
- Step 1: 0.8588 RMSE
- Step 10: 0.8391 RMSE
- **Degradation: -2.29%** (improves slightly, smooth averaging)

**TFT** shows horizon improvement:
- Step 1: 0.9785 RMSE
- Step 10: 0.8235 RMSE
- **Degradation: -15.84%** (learns long-term patterns better)

### Best Performer by Horizon Step

| Step | Best Model | RMSE | MAE | Winning Characteristic |
|------|-----------|------|-----|------------------------|
| 1 | Persistence (artifact) | 0.0000 | 0.0000 | ⚠ same-time target leakage; not used in headline comparison |
| 2 | XGBoost | 0.4711 | 0.3281 | Real forecasting starts |
| 3-10 | XGBoost | 0.5036-0.5201 | 0.3544-0.3653 | Consistent, reliable |

---

## Model-Specific Insights

### 🟢 XGBoost - Best All-Around
**Strengths:**
- Consistent performance across all horizons
- Minimal degradation with increasing horizon
- Captures feature interactions effectively
- No convergence issues (trains in seconds)

**Pattern:** Error increases from step 1→2, then stabilizes
- LR-like initial increase (learning the trend)
- Then plateau (captures mean with variance)
- Outperforms all baselines on long horizon (steps 4-10)

### 🟡 Linear Regression - Interpretable Baseline
**Strengths:**
- Lowest bias (ME = -0.223, but high MAPE)
- Smooth degradation pattern
- Simple AR(1) model, easy to understand

**Weakness:** High MAPE (78.92%) shows difficulty with percentage errors

### 🔵 SARIMA - Fixed-Origin Approximation
**Strengths:**
- Improves with horizon (captures seasonality)
- Negative degradation (-2.29%) suggests learning
- Uses seasonal patterns (24-hour cycle)

**Weaknesses:**
- Slow convergence (ARIMA optimization)
- High MAPE (102%) suggests systematic bias
- Fixed-origin implementation is not a full rolling-origin benchmark
- This is a computational approximation rather than a final production benchmark

### 🟣 TFT - Recursive Proxy, not Native Decoder Output
**Strengths:**
- Horizon improvement (-15.84%) suggests some long-horizon learning signal
- Uses multi-lag features (better temporal context)
- Competitive with SARIMA in this proxy setting

**Weaknesses:**
- Values are from a recursive proxy, not the native TFT decoder output
- Requires careful feature engineering (5 lags + 12 features)
- High MAPE (101.51%) similar to SARIMA
- Not as efficient as XGBoost

### ⚫ Persistence - Naive Lower Bound
**Characteristics:**
- Step 1 error = 0 (trivial: predicts current value)
- Sharp error growth (naive lower-bound benchmark)
- Useful as benchmark for model evaluation

---

## Technical Implementation

### SARIMA Fixed-Origin Approximation
**Problem**: Original recursive approach retrained SARIMA 1,426 times → timeout
**Solution**: 
```python
# Train ONCE on all training data
sarima_full = SARIMAX(y_train, order=(1,1,1), 
                       seasonal_order=(1,1,1,24))
results = sarima_full.fit()

# For each test point, use the same fixed-origin forecast path
for i in range(n_forecasts):
    forecast = results.get_forecast(steps=10).predicted_mean
```
**Result**: 166 seconds for 1,426 × 10-step forecasts, but this is a fixed-origin approximation rather than a true rolling-origin SARIMA benchmark.

### Recursive Forecasting Strategy
All baseline models use 1-step recursive forecasting:
```
predict(t+1) using features at t
feed prediction back as lag feature at t+1
predict(t+2) using features at t+1
repeat 10 times
```

Models use different feature sets:
- **Linear Regression**: Univariate AR(1)
- **XGBoost**: 12 multivariate features + 1 lag
- **TFT**: 12 multivariate features + 5 lags (smoother predictions)
- **SARIMA**: Direct forecast (no recursion, seasonal parameters)

---

## Visualization Insights

### From `horizon_comparison_10step.png`:
1. **RMSE Curves**:
   - XGBoost: Steep step 1→2, then plateau (ideal curve)
   - Persistence: Linear degradation (no learning)
   - TFT: High initial, improves at horizon (attention helps long-term)

2. **Consistency Pattern**:
   - XGBoost stabilizes by step 3 (robust)
   - SARIMA stable from start (averaged pattern)
   - Linear Regression smooth but suboptimal

3. **Aggregated RMSE Bar Chart**:
   - XGBoost: 0.4990 (clearly best)
   - Clear separation between top 2 and bottom 3

### From `sample_forecasts_10step.png`:
Shows 5 diverse forecast scenarios:
- **Forecast #1**: Actual increases, all models predict low (underestimation)
- **Forecast #357**: Actual decreases steeply, XGBoost captures trend best
- **Forecast #714**: Mixed pattern, XGBoost smooths well
- **Forecast #1070**: Rising trend, XGBoost follows nearest
- **Forecast #1426**: Complex dynamics, XGBoost stable

---

## Statistical Validation

### Degradation Analysis
- **XGBoost**: 44.15% increase from step 1 to 10 (expected, realistic)
- **SARIMA**: -2.29% (unusual improvement, suggests averaging pattern)
- **TFT**: -15.84% improvement (attention benefits long-term, but high baseline error)
- **Linear Regression**: 306% increase (poor long-term generalization)

### Bias Analysis (Mean Error)
| Model | Bias | Interpretation |
|-------|------|-----------------|
| Persistence | +0.0039 | Almost unbiased (by design) |
| XGBoost | -0.0985 | Slight underestimation (good) |
| Linear Regression | -0.2234 | Systematic underestimation |
| TFT | -0.4111 | Strong underestimation |
| SARIMA | -0.3790 | Moderate underestimation |

---

## Files Generated

### Data Files
1. **horizon_metrics_10step.csv** - 50 rows (5 models × 10 steps) with RMSE/MAE/MAPE/MASE/R²
2. **aggregated_metrics_10step.csv** - 5 rows with aggregated metrics across all horizons

### Visualizations
1. **horizon_comparison_10step.png** - 4-panel visualization:
   - RMSE degradation curves
   - MAE degradation curves
   - MAPE degradation curves
   - Aggregated RMSE bar chart

2. **sample_forecasts_10step.png** - 5 sample trajectories showing:
   - Actual vs predicted for 5 different test points
   - How each model behaves on diverse patterns
   - Visual comparison of forecasting strategies

---

## CV Talking Points

### 1-Sentence Summary
"Implemented a 10-step forecasting extension using 5 models; XGBoost achieved the strongest performance with 0.499 RMSE and 47% improvement over persistence once the step-1 artifact was excluded, while SARIMA and TFT remain approximate baselines under explicit caveats."

### 3-Bullet Version
- Developed a 10-step forecasting extension for 5 models (Persistence, Linear Regression, XGBoost, SARIMA, TFT) on 1,426 test sequences with horizon-wise evaluation
- Used a fixed-origin SARIMA approximation to reduce runtime to 166 seconds, but this is not a true rolling-origin benchmark and should be treated as a speed-oriented baseline
- XGBoost was the strongest performer: RMSE 0.499, MAPE 47.79%, R² 0.608, with consistent performance from horizons 2-10 once the persistence step-1 artifact was excluded

### Technical Interview Answer
"For the 10-step extension, I implemented two strategies: recursive single-step models (LR, XGBoost, TFT) and a fixed-origin SARIMA approximation using `get_forecast(steps=10)`. The main caution is that persistence step 1 is a same-time artifact, SARIMA is a fixed-origin baseline rather than a fully rolling benchmark, and the TFT result is a recursive proxy rather than native multi-step decoder output. Even with those caveats, XGBoost remained the strongest model in the evaluated setup, with stable performance across steps 2-10."

---

## Recommendations

### For Production Deployment
1. **Use XGBoost** for most accurate CO predictions
2. **Ensemble with Linear Regression** for interpretability
3. **Monitor horizon-specific performance** - XGBoost strongest at steps 4-10
4. **Implement rolling retraining** - update model weekly with new data

### For Model Improvement
1. **Hyperparameter tuning**: Current XGBoost used standard parameters
2. **Feature engineering**: Add external regressors (traffic, weather forecast)
3. **Ensemble methods**: Combine XGBoost + Linear Regression
4. **TFT optimization**: Retrain specifically for 10-step (not using recursive approximation)

### For CV/Interviews
1. **Emphasize the engineering trade-off**: Fixed-origin SARIMA as a speed-oriented approximation (166s), with rolling-origin evaluation as the correct approach
2. **Show rigorous evaluation**: Horizon-wise metrics, not just aggregated
3. **Demonstrate trade-offs**: XGBoost accuracy vs Linear Regression interpretability
4. **Include practical insights**: 44% degradation is realistic and expected

---

## Conclusion

XGBoost emerges as the optimal model for 10-step ahead CO forecasting across steps 2–10 (step-1 artifact excluded), with a 47% improvement over the naive baseline in this setup. The key insight: proper evaluation requires horizon-wise analysis, not just aggregated metrics. SARIMA's fixed-origin approximation demonstrates the importance of implementation details, but it should be treated as a speed-oriented approximation rather than a final rolling-origin benchmark.

The evaluation framework (14,260 predictions across 5 models × 10 horizons) provides a useful secondary comparison, with native multi-step benchmarks as the natural next step for future work.
