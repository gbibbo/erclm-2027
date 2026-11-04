## S, symmetric prompting, points, B = 2000

| backbone | quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- | --- |
| base | duration AUROC original | 86.65 | 83.84 | 89.2 | 1.37 |
| base | duration AUROC sym2 | 88.77 | 85.96 | 91.31 | 1.35 |
| base | duration AUROC sym4 | 88.76 | 85.98 | 91.26 | 1.34 |
| base | duration BA original L0 | 50.08 | 50.0 | 50.21 | 0.06 |
| base | duration BA original L1 | 77.96 | 75.71 | 80.33 | 1.19 |
| base | duration BA sym2 | 55.75 | 54.88 | 56.71 | 0.46 |
| base | duration BA sym4 | 56.29 | 55.25 | 57.33 | 0.53 |
| base | duration best BA sym2 | 84.46 | 82.42 | 87.33 | 1.29 |
| base | duration operating-point loss sym2 | 28.71 | 26.58 | 31.58 | 1.31 |
| base | duration BA sym2 - original L1 | -22.21 | -24.75 | -20.0 | 1.22 |
| base | duration BA sym2 - original L0 | 5.67 | 4.79 | 6.58 | 0.46 |
| base | duration BA sym2 - LoRA L0 (original format) | -27.96 | -30.29 | -25.67 | 1.2 |
| base | duration AUROC sym2 - original | 2.12 | 1.05 | 3.26 | 0.58 |
| base | duration BA sym4 - original L1 | -21.67 | -24.21 | -19.29 | 1.25 |
| base | snr AUROC original | 91.96 | 89.2 | 94.39 | 1.32 |
| base | snr AUROC sym2 | 96.17 | 94.41 | 97.64 | 0.85 |
| base | snr AUROC sym4 | 96.56 | 94.93 | 97.94 | 0.78 |
| base | snr BA original L0 | 50.06 | 50.0 | 50.17 | 0.05 |
| base | snr BA original L1 | 81.78 | 78.72 | 84.5 | 1.51 |
| base | snr BA sym2 | 65.22 | 62.89 | 67.78 | 1.24 |
| base | snr BA sym4 | 67.0 | 64.5 | 69.61 | 1.32 |
| base | snr best BA sym2 | 91.61 | 89.67 | 94.17 | 1.16 |
| base | snr operating-point loss sym2 | 26.39 | 23.78 | 29.83 | 1.52 |
| base | snr BA sym2 - original L1 | -16.56 | -19.67 | -13.22 | 1.69 |
| base | snr BA sym2 - original L0 | 15.17 | 12.83 | 17.72 | 1.24 |
| base | snr BA sym2 - LoRA L0 (original format) | -31.5 | -34.22 | -28.5 | 1.45 |
| base | snr AUROC sym2 - original | 4.21 | 2.81 | 5.83 | 0.75 |
| base | snr BA sym4 - original L1 | -14.78 | -17.89 | -11.28 | 1.7 |
| lora | duration AUROC original | 91.9 | 89.4 | 94.11 | 1.2 |
| lora | duration AUROC sym2 | 91.69 | 89.13 | 93.94 | 1.22 |
| lora | duration AUROC sym4 | 91.74 | 89.18 | 93.97 | 1.22 |
| lora | duration BA original L0 | 83.71 | 81.46 | 85.88 | 1.13 |
| lora | duration BA original L1 | 84.75 | 82.58 | 86.92 | 1.1 |
| lora | duration BA sym2 | 84.67 | 82.42 | 86.92 | 1.14 |
| lora | duration BA sym4 | 84.42 | 82.21 | 86.62 | 1.13 |
| lora | duration best BA sym2 | 87.58 | 86.0 | 90.38 | 1.13 |
| lora | duration operating-point loss sym2 | 2.92 | 2.37 | 5.04 | 0.68 |
| lora | duration BA sym2 - original L1 | -0.08 | -0.67 | 0.46 | 0.29 |
| lora | duration BA sym2 - original L0 | 0.96 | 0.37 | 1.58 | 0.32 |
| lora | duration BA sym2 - LoRA L0 (original format) | 0.96 | 0.37 | 1.58 | 0.32 |
| lora | duration AUROC sym2 - original | -0.21 | -0.69 | 0.23 | 0.24 |
| lora | duration BA sym4 - original L1 | -0.33 | -1.0 | 0.25 | 0.31 |
| lora | snr AUROC original | 98.85 | 97.64 | 99.69 | 0.52 |
| lora | snr AUROC sym2 | 99.01 | 97.85 | 99.77 | 0.51 |
| lora | snr AUROC sym4 | 99.0 | 97.81 | 99.76 | 0.52 |
| lora | snr BA original L0 | 96.72 | 94.83 | 98.22 | 0.87 |
| lora | snr BA original L1 | 96.78 | 94.89 | 98.28 | 0.86 |
| lora | snr BA sym2 | 96.39 | 94.44 | 98.06 | 0.91 |
| lora | snr BA sym4 | 96.44 | 94.5 | 98.11 | 0.92 |
| lora | snr best BA sym2 | 97.67 | 96.28 | 98.94 | 0.69 |
| lora | snr operating-point loss sym2 | 1.28 | 0.56 | 2.56 | 0.51 |
| lora | snr BA sym2 - original L1 | -0.39 | -1.17 | 0.22 | 0.36 |
| lora | snr BA sym2 - original L0 | -0.33 | -1.11 | 0.28 | 0.35 |
| lora | snr BA sym2 - LoRA L0 (original format) | -0.33 | -1.11 | 0.28 | 0.35 |
| lora | snr AUROC sym2 - original | 0.17 | -0.12 | 0.5 | 0.16 |
| lora | snr BA sym4 - original L1 | -0.33 | -1.06 | 0.22 | 0.34 |
