# Clarifying Notes for the 10-Step Forecasting Evaluation

## Summary

This document is an exploratory note for the 10-step extension only. It is not the official final evaluation of the project.

The official, validated project claim remains the single-step benchmarking results: XGBoost was the strongest model for the primary forecasting task. The 10-step results below are included as a secondary analysis and should be described with explicit caveats.

The 10-step extension contains three important methodological caveats. They do not invalidate the broader workflow, but they do affect how the results should be described in a report, CV, or interview.

The key point is this: the strongest conclusion that is still defensible is that XGBoost was the best among the recursive baseline models evaluated under the current 10-step setup, while the persistence, SARIMA, and TFT results should be interpreted with clear caveats.

---

## 1) Persistence Step 1 is an indexing artifact, not a real performance result

The issue is real.

A true persistence forecast for horizon 1 is:

- predicted at t+1 = observed at t

That means the first target should be the value at time t+1, not the value at time t. In the notebook, the evaluation was effectively comparing:

- prediction at t
- target at t

which makes step 1 appear to have zero error by construction.

This is why the reported output showed:

- Persistence RMSE = 0.0000 at Step 1
- Persistence R² = 1.0000 at Step 1

Those values are impossible for a valid one-step persistence model on a non-constant series.

### Correct interpretation

Either:

1. fix the indexing so the forecast origin is t and the target is t+1, or
2. explicitly exclude step 1 from the persistence benchmark and note that persistence is evaluated from step 2 onward.

The correct wording is:

> Persistence was evaluated as a naive baseline using the last observed value as the forecast for future horizons, with step 1 omitted or re-indexed to avoid the tautological same-time comparison.

This should be reflected in any CV or report language.

---

## 2) SARIMA as implemented is not a true pointwise rolling-origin forecast

The concern is also valid.

In the notebook, SARIMA was fit once on the whole training set, and then the same forecast object was reused for each test starting point:

- `sarima_full.fit(...)`
- then repeated `results_sarima_full.get_forecast(steps=10)` across many origins

That produces the same 10-step forecast from the same end-of-training origin, not a sequence of forecasts that update with newly observed data.

This means the model was effectively behaving like:

- one fitted SARIMA model
- one common forecast path copied across 1,426 origin points

rather than a true rolling-origin evaluation.

### Why this matters

A genuine multi-step SARIMA evaluation should use a rolling-origin or expanding-window procedure, for example:

- append the next observed value,
- refit or update the model without refitting the whole process,
- then forecast the next 10 steps,
- repeat.

### Correct interpretation

The SARIMA result in the current notebook should be described as:

> a computationally efficient SARIMA baseline using a fixed-origin forecast, not a fully updated rolling-origin SARIMA benchmark.

If the goal is to keep SARIMA in the comparison, it should be rerun properly with incremental updates or omitted from the headline ranking until that is done.

---

## 3) TFT was not evaluated natively; it was approximated recursively

This is a major caveat and the report should say so plainly.

The current TFT setup was not a native 10-step decoder prediction from a trained TFT model. Instead, it was approximated by:

- building a recursive feature set,
- training a Ridge regressor on lagged features,
- projecting that as a TFT-style multi-step forecast.

This is useful as a proxy, but it is not the same as evaluating the actual TFT decoder output trained for multi-horizon forecasting.

### Correct interpretation

The TFT result should be labeled as:

> a recursive proxy evaluation, not a native TFT 10-step forecast.

This is consistent with the project’s own warning that the model should ideally be retrained specifically for 10-step forecasting using its native decoder structure.

If a native TFT decoder is available, that should be used before presenting TFT as a valid competitor in the final results.

---

## What remains defensible

Even with these caveats, the following still holds:

- XGBoost was the strongest among the recursive multi-step baseline models evaluated under the current setup.
- The notebook framework and comparison logic are valuable for model benchmarking.
- The main weakness is not the overall workflow, but the way the persistence, SARIMA, and TFT baselines were framed.

The correct presentation is therefore not "all models are equally valid 10-step benchmarks" but rather:

> XGBoost performed best among the evaluated recursive forecasting baselines; however, the persistence benchmark should be re-indexed, the SARIMA benchmark should be validated as a rolling-origin baseline, and the TFT result should be treated as a proxy until the native TFT decoder output is used.

---

## Recommended wording for the final report or CV

### Safe version

> Evaluated multistep forecasting performance for CO prediction using recursive baselines and a seasonal ARIMA comparator. XGBoost was the strongest among the baseline models evaluated, while persistence was corrected for an off-by-one indexing issue and SARIMA/TFT were treated with explicit caveats because they were not fully native multi-step benchmarks.

### Interview-safe version

> Compared forecasting baselines for hourly CO prediction, with special attention to recursive validation and forecast origin alignment. XGBoost delivered the strongest performance among the evaluated models; however, the final narrative distinguishes between native multi-step forecasting and proxy baselines for persistence, SARIMA, and TFT.

---

## Final recommendation

Before publishing the results as a final technical claim, the following should be done:

1. Re-index the persistence benchmark so step 1 uses t+1 as the target, or exclude step 1.
2. Label SARIMA as a fixed-origin or approximate baseline unless a proper rolling-origin re-run is performed.
3. Recast TFT as a recursive proxy until native 10-step TFT output is available.
4. Keep the headline winner only as "best among evaluated recursive baselines" rather than a fully universal claim across all native multi-step models.

That keeps the story honest and interview-ready without undermining the core modeling work.
