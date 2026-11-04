## V causes, clean provenance bank (76 recordings), post hoc, 26 October

| duration ms | AUROC ab | AUROC ba_speech_second | AUROC recipe | AUROC LoRA | corr within non-speech | corr within speech | recipe L1 TPR | recipe L1 TNR | LoRA L0 TPR | LoRA L0 TNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 82.6 | 72.33 | 82.26 | 87.79 | 0.27 | 0.59 | 56.58 | 88.16 | 14.47 | 100.0 |
| 40 | 88.76 | 80.07 | 86.24 | 91.86 | 0.45 | 0.78 | 57.89 | 92.11 | 46.05 | 98.68 |
| 60 | 90.82 | 87.26 | 89.77 | 93.14 | 0.67 | 0.74 | 72.37 | 94.74 | 68.42 | 96.05 |
| 80 | 91.69 | 88.18 | 91.35 | 94.61 | 0.67 | 0.68 | 72.37 | 93.42 | 72.37 | 96.05 |
| 100 | 94.5 | 92.62 | 94.09 | 96.59 | 0.6 | 0.55 | 81.58 | 93.42 | 81.58 | 98.68 |
| 200 | 93.2 | 87.84 | 93.09 | 96.73 | 0.46 | 0.47 | 84.21 | 92.11 | 81.58 | 98.68 |
| 500 | 95.84 | 84.06 | 94.05 | 99.39 | 0.3 | 0.5 | 81.58 | 88.16 | 93.42 | 100.0 |
| 1000 | 92.96 | 84.21 | 92.27 | 99.91 | 0.41 | 0.4 | 77.63 | 90.79 | 96.05 | 100.0 |

means: {'AUROC ab': 91.3, 'AUROC ba_speech_second': 84.57, 'AUROC recipe': 90.39, 'AUROC LoRA': 95.0, 'corr within non-speech': 0.48, 'corr within speech': 0.59, 'recipe L1 TPR': 73.03, 'recipe L1 TNR': 91.61, 'LoRA L0 TPR': 69.24, 'LoRA L0 TNR': 98.52}

## Does batch calibration fail on the LoRA because its probabilities saturate? Whole-batch thresholds, post hoc

| bank | ms | LoRA L0 | LoRA L1 | LoRA P1 | LoRA best | L1 threshold | P1 threshold | median s speech | median s non-speech | share of non-speech with sigmoid < 0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| provenance, 76 clean, balanced | 20.0 | 57.24 | 61.84 | 58.55 | 82.89 | -2.43 | -1.26 | -4.77 | -6.72 | 94.74 |
| provenance, 76 clean, balanced | 40.0 | 72.37 | 75.0 | 75.66 | 87.5 | -1.15 | -1.36 | -0.63 | -6.61 | 89.47 |
| provenance, 76 clean, balanced | 100.0 | 90.13 | 90.13 | 92.11 | 94.74 | -0.39 | -1.27 | 3.54 | -6.71 | 88.16 |
| provenance, 76 clean, balanced | 1000.0 | 98.03 | 98.03 | 98.68 | 99.34 | -0.11 | -0.62 | 5.46 | -6.8 | 90.79 |
| main, 470 held out, 29% speech | 20.0 | 60.29 | 64.14 | 61.4 | 73.07 | -2.56 | -1.52 | -4.52 | -6.23 | 90.45 |
| main, 470 held out, 29% speech | 40.0 | 71.63 | 74.06 | 73.25 | 80.11 | -1.78 | -1.32 | -1.72 | -6.09 | 81.49 |
| main, 470 held out, 29% speech | 100.0 | 86.51 | 88.66 | 88.66 | 91.24 | -1.2 | -1.11 | 3.08 | -6.17 | 81.49 |
| main, 470 held out, 29% speech | 1000.0 | 99.11 | 99.18 | 99.55 | 99.55 | -0.94 | -1.08 | 5.7 | -7.41 | 94.33 |

## LoRA on the balanced clean provenance batch, label-free thresholds in logit space, post hoc

| ms | L1 (mean probability) | mean of logits | median of logits | best |
| --- | --- | --- | --- | --- |
| 20 | 61.84 | 73.68 | 82.89 | 82.89 |
| 40 | 75.0 | 78.29 | 85.53 | 87.5 |
| 60 | 81.58 | 84.21 | 89.47 | 90.13 |
| 80 | 85.53 | 85.53 | 90.79 | 91.45 |
| 100 | 90.13 | 92.11 | 94.74 | 94.74 |
| 200 | 90.13 | 92.11 | 93.42 | 94.74 |
| 500 | 96.71 | 96.05 | 96.05 | 96.71 |
| 1000 | 98.03 | 98.68 | 98.68 | 99.34 |

means: {'L1 (mean probability)': 84.87, 'mean of logits': 87.58, 'median of logits': 91.45, 'best': 92.19}

## Whole main bank, 970 clips, balanced, duration axis, thresholds on each whole condition batch, post hoc

| ms | recipe L1 | recipe median | recipe best | LoRA L1 | LoRA median | LoRA best | LoRA L0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 69.9 | 65.8 | 70.4 | 64.0 | 69.5 | 71.4 | 61.9 |
| 40 | 77.8 | 74.3 | 78.0 | 74.2 | 78.1 | 79.6 | 72.7 |
| 60 | 82.4 | 81.4 | 83.0 | 81.9 | 81.8 | 83.2 | 81.5 |
| 80 | 84.3 | 83.2 | 85.3 | 83.5 | 85.6 | 87.0 | 82.5 |
| 100 | 86.3 | 86.1 | 86.8 | 87.1 | 88.0 | 89.3 | 85.8 |
| 200 | 88.5 | 89.3 | 89.9 | 93.8 | 93.0 | 94.1 | 93.3 |
| 500 | 88.9 | 91.1 | 91.9 | 97.7 | 97.9 | 98.0 | 97.6 |
| 1000 | 89.1 | 90.3 | 91.1 | 98.1 | 98.6 | 98.6 | 98.0 |

means: {'recipe L1': 83.39, 'recipe median': 82.69, 'recipe best': 84.55, 'LoRA L1': 85.05, 'LoRA median': 86.56, 'LoRA best': 87.65, 'LoRA L0': 84.16}

## Clean provenance bank, 76 recordings, balanced, whole condition batch, post hoc

| ms | recipe L1 | recipe median | recipe best | LoRA L1 | LoRA median | LoRA best | LoRA L0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20 | 72.4 | 77.6 | 78.9 | 61.8 | 82.9 | 82.9 | 57.2 |
| 40 | 75.0 | 77.6 | 80.3 | 75.0 | 85.5 | 87.5 | 72.4 |
| 60 | 83.6 | 85.5 | 86.2 | 81.6 | 89.5 | 90.1 | 82.2 |
| 80 | 82.9 | 84.2 | 86.8 | 85.5 | 90.8 | 91.4 | 84.2 |
| 100 | 87.5 | 89.5 | 90.8 | 90.1 | 94.7 | 94.7 | 90.1 |
| 200 | 88.2 | 86.8 | 88.8 | 90.1 | 93.4 | 94.7 | 90.1 |
| 500 | 84.9 | 86.8 | 87.5 | 96.7 | 96.1 | 96.7 | 96.7 |
| 1000 | 84.2 | 85.5 | 87.5 | 98.0 | 98.7 | 99.3 | 98.0 |

means: {'recipe L1': 82.32, 'recipe median': 84.21, 'recipe best': 85.86, 'LoRA L1': 84.87, 'LoRA median': 91.45, 'LoRA best': 92.19, 'LoRA L0': 83.88}
