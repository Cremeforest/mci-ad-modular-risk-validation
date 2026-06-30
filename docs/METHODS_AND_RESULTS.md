# Methods and results summary

## Pipeline

1. Construct longitudinal MCI clinical histories.
2. Build 1/2/3/5-year conversion labels.
3. Train the final summary-augmented modular longitudinal model.
4. Audit uncertainty and calibration on the internal test split.
5. Externally validate the frozen ADNI-trained model on NACC.
6. Audit external calibration and local recalibration under cohort shift.

## Final model design

The model uses one module per clinical feature plus a visit-process module. Each feature module combines observed-visit temporal encoding with module-local longitudinal summaries such as last value, mean, variability, change, slope, observed count, and observed rate.

## Internal performance

| Horizon | AUROC | AUPRC | Brier |
| --- | --- | --- | --- |
| 1y | 0.916 (0.880-0.949) | 0.412 (0.287-0.573) | 0.047 (0.036-0.058) |
| 2y | 0.904 (0.869-0.933) | 0.585 (0.483-0.718) | 0.085 (0.071-0.100) |
| 3y | 0.914 (0.880-0.943) | 0.822 (0.746-0.877) | 0.099 (0.081-0.118) |
| 5y | 0.926 (0.899-0.952) | 0.903 (0.860-0.940) | 0.108 (0.088-0.127) |


## NACC external validation

| Horizon | AUROC | AUPRC | Brier |
| --- | --- | --- | --- |
| 1y | 0.733 (0.720-0.746) | 0.298 (0.279-0.318) | 0.122 |
| 2y | 0.740 (0.729-0.751) | 0.529 (0.509-0.549) | 0.192 |
| 3y | 0.740 (0.727-0.750) | 0.664 (0.646-0.681) | 0.241 |
| 5y | 0.749 (0.735-0.762) | 0.835 (0.822-0.848) | 0.286 |


## External calibration finding

Raw frozen probabilities underestimated long-horizon absolute risk in NACC. Cross-fitted NACC local Platt recalibration improved calibration without retraining the prediction model.

## Recommended wording

The frozen ADNI-trained model retained moderate external discrimination on the NACC first-MCI cohort under a no-ADAS13 scenario. Raw frozen absolute risks were underestimated under cross-cohort shift, whereas cross-fitted local Platt recalibration substantially improved calibration without retraining the prediction model.
