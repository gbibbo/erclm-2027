## Methods on the 470 held-out clips, recipe = average of ('ab', 'ba_speech_second'), points, B = 2000

{'frozen thresholds': {'original': -3.13671875, 'recipe': -1.458984375}, 'held-out clips': 470} 

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration | original | AUROC | 89.38 | 87.42 | 91.19 | 0.97 |
| duration | original | best BA | 83.29 | 81.87 | 85.83 | 1.01 |
| duration | original | BA L0 | 50.0 | 50.0 | 50.0 | 0.0 |
| duration | original | BA L1 | 81.99 | 79.83 | 84.29 | 1.12 |
| duration | original | BA frozen | 80.77 | 78.67 | 82.75 | 1.02 |
| duration | recipe | AUROC | 91.41 | 89.55 | 93.14 | 0.93 |
| duration | recipe | best BA | 85.96 | 84.16 | 88.4 | 1.08 |
| duration | recipe | BA L0 | 56.19 | 55.06 | 57.49 | 0.61 |
| duration | recipe | BA L1 | 84.82 | 82.35 | 86.81 | 1.15 |
| duration | recipe | BA frozen | 82.76 | 80.61 | 84.89 | 1.11 |
| duration | LoRA | AUROC | 94.03 | 92.39 | 95.5 | 0.8 |
| duration | LoRA | best BA | 89.0 | 87.6 | 91.27 | 0.94 |
| duration | LoRA | BA L0 | 84.55 | 82.34 | 86.8 | 1.13 |
| duration | LoRA | BA L1 | 86.04 | 83.94 | 87.94 | 1.07 |
| duration | recipe | BA P1 | 83.7 | 81.24 | 85.64 | 1.11 |
| duration | recipe L1 - LoRA L0 | 0.26 | -1.73 | 1.99 | 0.96 |
| duration | recipe P1 - LoRA L0 | -0.85 | -2.54 | 0.53 | 0.79 |
| duration | recipe frozen - LoRA L0 | -1.8 | -3.83 | 0.4 | 1.1 |
| duration | recipe L1 - LoRA L1 | -1.22 | -2.97 | 0.33 | 0.85 |
| duration | recipe P1 - LoRA L1 | -2.34 | -3.85 | -1.14 | 0.7 |
| duration | recipe L1 - original L1 | 2.82 | 0.93 | 4.17 | 0.83 |
| duration | recipe L0 - original L0 | 6.19 | 5.06 | 7.49 | 0.61 |
| duration | original L1 - original L0 | 31.99 | 29.83 | 34.29 | 1.12 |
| duration | AUROC recipe - original | 2.03 | 1.05 | 3.07 | 0.52 |
| duration | AUROC recipe - LoRA | -2.62 | -3.79 | -1.53 | 0.58 |
| duration | best BA recipe - LoRA L0 | 1.4 | 0.13 | 3.6 | 0.88 |
| duration | best BA recipe - best BA LoRA | -3.05 | -4.51 | -1.68 | 0.72 |
| snr | original | AUROC | 92.51 | 90.43 | 94.28 | 0.99 |
| snr | original | best BA | 87.6 | 85.42 | 90.14 | 1.21 |
| snr | original | BA L0 | 50.0 | 50.0 | 50.0 | 0.0 |
| snr | original | BA L1 | 86.2 | 83.23 | 88.63 | 1.37 |
| snr | original | BA frozen | 86.33 | 83.69 | 88.59 | 1.28 |
| snr | recipe | AUROC | 96.33 | 95.04 | 97.42 | 0.59 |
| snr | recipe | best BA | 92.78 | 91.59 | 94.57 | 0.75 |
| snr | recipe | BA L0 | 65.16 | 62.61 | 67.79 | 1.33 |
| snr | recipe | BA L1 | 91.6 | 90.06 | 93.09 | 0.74 |
| snr | recipe | BA frozen | 89.68 | 88.0 | 91.31 | 0.84 |
| snr | LoRA | AUROC | 99.77 | 99.57 | 99.91 | 0.09 |
| snr | LoRA | best BA | 98.76 | 98.32 | 99.41 | 0.28 |
| snr | LoRA | BA L0 | 98.0 | 97.21 | 98.72 | 0.39 |
| snr | LoRA | BA L1 | 97.71 | 96.86 | 98.5 | 0.43 |
| snr | recipe | BA P1 | 90.49 | 88.82 | 92.39 | 0.91 |
| snr | recipe L1 - LoRA L0 | -6.39 | -7.68 | -5.08 | 0.68 |
| snr | recipe P1 - LoRA L0 | -7.51 | -9.04 | -5.76 | 0.84 |
| snr | recipe frozen - LoRA L0 | -8.32 | -9.82 | -6.82 | 0.76 |
| snr | recipe L1 - LoRA L1 | -6.11 | -7.41 | -4.82 | 0.67 |
| snr | recipe P1 - LoRA L1 | -7.22 | -8.71 | -5.47 | 0.83 |
| snr | recipe L1 - original L1 | 5.41 | 3.74 | 7.5 | 0.96 |
| snr | recipe L0 - original L0 | 15.16 | 12.61 | 17.79 | 1.33 |
| snr | original L1 - original L0 | 36.2 | 33.23 | 38.63 | 1.37 |
| snr | AUROC recipe - original | 3.83 | 2.74 | 5.13 | 0.62 |
| snr | AUROC recipe - LoRA | -3.43 | -4.71 | -2.38 | 0.58 |
| snr | best BA recipe - LoRA L0 | -5.22 | -6.25 | -3.61 | 0.67 |
| snr | best BA recipe - best BA LoRA | -5.98 | -7.24 | -4.44 | 0.71 |
