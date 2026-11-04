## Checks of a reviewer's claims on the 470 held-out clips (post hoc, B = 2000)

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration LoRA AUROC | 94.03 | 92.39 | 95.5 | 0.8 |
| duration recipe AUROC | 91.41 | 89.55 | 93.14 | 0.93 |
| duration AUROC recipe - LoRA | -2.62 | -3.79 | -1.53 | 0.58 |
| duration LoRA best BA | 89.0 | 87.6 | 91.27 | 0.94 |
| duration recipe best BA | 85.96 | 84.16 | 88.4 | 1.08 |
| duration LoRA L1 BA | 86.04 | 83.94 | 87.94 | 1.07 |
| duration recipe L1 - LoRA L1 | -1.22 | -2.97 | 0.33 | 0.85 |
| duration recipe pooled-batch BA - LoRA L0 | -0.59 | -2.31 | 1.16 | 0.88 |
| duration recipe separate-batch BA - LoRA L0 | -5.13 | -6.6 | -3.58 | 0.76 |
| snr LoRA AUROC | 99.77 | 99.57 | 99.91 | 0.09 |
| snr recipe AUROC | 96.33 | 95.04 | 97.42 | 0.59 |
| snr AUROC recipe - LoRA | -3.43 | -4.71 | -2.38 | 0.58 |
| snr LoRA best BA | 98.76 | 98.32 | 99.41 | 0.28 |
| snr recipe best BA | 92.78 | 91.59 | 94.57 | 0.75 |
| snr LoRA L1 BA | 97.71 | 96.86 | 98.5 | 0.43 |
| snr recipe L1 - LoRA L1 | -6.11 | -7.41 | -4.82 | 0.67 |
| snr recipe pooled-batch BA - LoRA L0 | -6.37 | -7.76 | -4.95 | 0.7 |
| snr recipe separate-batch BA - LoRA L0 | -12.09 | -14.68 | -9.53 | 1.32 |

Batch calibration with a different speech share in the calibration batch (duration axis)

| speech share of the calibration batch | recipe BA on duration | 2.5% | 97.5% |
| --- | --- | --- | --- |
| 0.1 | 80.3 | 80.03 | 80.63 |
| 0.29 | 84.82 | 84.82 | 84.82 |
| 0.5 | 84.64 | 84.47 | 84.8 |

Per-condition thresholds calibrated once on the 300 discovery clips (unlabeled, balanced), applied clip by clip to the 470 held-out clips

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration recipe, per-condition thresholds from the separate unlabeled clips - LoRA L0 | -3.1 | -4.84 | -1.35 | 0.89 |
| snr recipe, per-condition thresholds from the separate unlabeled clips - LoRA L0 | -12.2 | -14.78 | -9.61 | 1.33 |
