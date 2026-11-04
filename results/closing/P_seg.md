## P, calibration that should not depend on the speech share, on the average of ('ab_speech_second', 'ba_speech_second'), BA points, B = 2000

{'held-out clips': 470, 'P1 failed fits in the sweep': 36, 'Platt map (slope, intercept)': (1.53819719596183, 2.33412418162541), 'training speech share': 0.5, 'fixed-batch speech share estimates': {'300 prompt-control clips, 50% speech': {'P1': 0.38099057651630336, 'P2': 0.9999990000000002}, '500 discovery clips, 70% speech': {'P1': 0.5159534652618418, 'P2': 0.9999990000000002}}} 

### Reading of the sweep

|  | range | lowest | L1 at 50% | flat (range <= 2) | no cost (lowest >= L1 at 50% - 2) |
| --- | --- | --- | --- | --- | --- |
| duration | L1 | 8.619720840243232 | 74.90110903814262 | 82.47584300718628 | False | False |
| duration | P1 | 4.279971669430623 | 76.13415561083471 | 82.47584300718628 | False | False |
| duration | P1 own variances | 3.9314123825317893 | 78.45096047540078 | 82.47584300718628 | False | False |
| duration | P2 | 8.017886263128801 | 75.65241846323936 | 82.47584300718628 | False | False |
| snr | L1 | 14.478989312695774 | 79.0712962962963 | 92.47874055647688 | False | False |
| snr | P1 | 0.9589874700571386 | 92.12727565874332 | 92.47874055647688 | True | True |
| snr | P1 own variances | 4.483107610097662 | 87.612350285609 | 92.47874055647688 | False | False |
| snr | P2 | 7.7320803390454955 | 85.21422517044408 | 92.47874055647688 | False | False |

### Sweep, 150 calibration clips from the discovery recordings, the 200 draws of g1t

| axis | method | speech share | BA | 2.5% | 97.5% | BA - LoRA L0 | estimated share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| duration | L1 | 0.1 | 80.16 | 78.9 | 81.03 | -4.39 | nan |
| duration | P1 | 0.1 | 80.41 | 78.49 | 82.1 | -4.14 | 0.14 |
| duration | P1 own variances | 0.1 | 81.36 | 79.81 | 82.72 | -3.19 | nan |
| duration | P2 | 0.1 | 75.65 | 74.92 | 76.34 | -8.9 | 0.85 |
| snr | L1 | 0.1 | 88.32 | 87.54 | 89.11 | -9.68 | nan |
| snr | P1 | 0.1 | 93.02 | 92.02 | 93.88 | -4.98 | 0.14 |
| snr | P1 own variances | 0.1 | 89.74 | 87.19 | 91.72 | -8.26 | nan |
| snr | P2 | 0.1 | 85.74 | 85.21 | 86.23 | -12.26 | 0.85 |
| duration | L1 | 0.2 | 82.46 | 81.9 | 83.04 | -2.1 | nan |
| duration | P1 | 0.2 | 79.88 | 78.19 | 81.54 | -4.67 | 0.18 |
| duration | P1 own variances | 0.2 | 82.13 | 80.74 | 83.19 | -2.43 | nan |
| duration | P2 | 0.2 | 78.48 | 77.16 | 79.61 | -6.08 | 0.94 |
| snr | L1 | 0.2 | 91.72 | 91.19 | 92.25 | -6.28 | nan |
| snr | P1 | 0.2 | 93.08 | 92.33 | 93.64 | -4.92 | 0.18 |
| snr | P1 own variances | 0.2 | 91.23 | 89.0 | 92.46 | -6.77 | nan |
| snr | P2 | 0.2 | 86.95 | 86.44 | 87.54 | -11.05 | 0.94 |
| duration | L1 | 0.3 | 83.52 | 82.93 | 84.12 | -1.03 | nan |
| duration | P1 | 0.3 | 79.67 | 78.08 | 81.21 | -4.88 | 0.24 |
| duration | P1 own variances | 0.3 | 82.38 | 80.66 | 83.41 | -2.17 | nan |
| duration | P2 | 0.3 | 80.95 | 79.97 | 81.77 | -3.61 | 0.98 |
| snr | L1 | 0.3 | 93.55 | 93.28 | 93.76 | -4.45 | nan |
| snr | P1 | 0.3 | 93.09 | 92.59 | 93.58 | -4.91 | 0.24 |
| snr | P1 own variances | 0.3 | 91.64 | 89.08 | 92.98 | -6.36 | nan |
| snr | P2 | 0.3 | 89.01 | 88.31 | 89.81 | -8.99 | 0.98 |
| duration | L1 | 0.5 | 82.48 | 81.81 | 83.27 | -2.08 | nan |
| duration | P1 | 0.5 | 79.07 | 77.78 | 80.53 | -5.48 | 0.37 |
| duration | P1 own variances | 0.5 | 82.2 | 80.44 | 83.53 | -2.35 | nan |
| duration | P2 | 0.5 | 83.67 | 83.09 | 84.14 | -0.88 | 1.0 |
| snr | L1 | 0.5 | 92.48 | 91.78 | 92.98 | -5.52 | nan |
| snr | P1 | 0.5 | 92.96 | 92.36 | 93.48 | -5.04 | 0.37 |
| snr | P1 own variances | 0.5 | 92.1 | 90.0 | 93.38 | -5.9 | nan |
| snr | P2 | 0.5 | 92.95 | 92.34 | 93.45 | -5.05 | 1.0 |
| duration | L1 | 0.7 | 79.57 | 78.25 | 80.5 | -4.98 | nan |
| duration | P1 | 0.7 | 78.16 | 75.78 | 80.13 | -6.39 | 0.51 |
| duration | P1 own variances | 0.7 | 81.25 | 78.28 | 83.07 | -3.3 | nan |
| duration | P2 | 0.7 | 82.48 | 81.53 | 83.42 | -2.08 | 1.0 |
| snr | L1 | 0.7 | 87.73 | 86.11 | 88.85 | -10.27 | nan |
| snr | P1 | 0.7 | 92.81 | 92.06 | 93.39 | -5.19 | 0.51 |
| snr | P1 own variances | 0.7 | 91.25 | 86.9 | 93.47 | -6.75 | nan |
| snr | P2 | 0.7 | 92.71 | 92.03 | 93.27 | -5.29 | 1.0 |
| duration | L1 | 0.9 | 74.9 | 73.62 | 76.45 | -9.65 | nan |
| duration | P1 | 0.9 | 76.13 | 73.2 | 78.9 | -8.42 | 0.64 |
| duration | P1 own variances | 0.9 | 78.45 | 74.07 | 81.3 | -6.1 | nan |
| duration | P2 | 0.9 | 78.81 | 77.43 | 79.8 | -5.74 | 1.0 |
| snr | L1 | 0.9 | 79.07 | 77.85 | 80.56 | -18.93 | nan |
| snr | P1 | 0.9 | 92.13 | 90.02 | 93.29 | -5.87 | 0.64 |
| snr | P1 own variances | 0.9 | 87.61 | 82.69 | 92.28 | -10.39 | nan |
| snr | P2 | 0.9 | 85.21 | 83.1 | 87.35 | -12.79 | 1.0 |

### Paired bootstrap on the 470 held-out clips

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration | test batch | P1 - LoRA L0 | -4.83 | -6.64 | -3.13 | 0.9 |
| duration | test batch | P1 - L1 | -3.64 | -5.58 | -1.97 | 0.93 |
| duration | test batch | P2 - LoRA L0 | -4.1 | -5.97 | -1.99 | 1.03 |
| duration | test batch | P2 - L1 | -2.9 | -4.09 | -1.58 | 0.65 |
| duration | test batch | L1 - LoRA L0 | -1.2 | -2.97 | 0.68 | 0.91 |
| duration | 300 prompt-control clips, 50% speech | P1 - LoRA L0 | -5.16 | -6.6 | -3.67 | 0.74 |
| duration | 300 prompt-control clips, 50% speech | P1 - L1 | -3.11 | -4.14 | -2.08 | 0.53 |
| duration | 300 prompt-control clips, 50% speech | P2 - LoRA L0 | -0.9 | -2.77 | 1.02 | 0.97 |
| duration | 300 prompt-control clips, 50% speech | P2 - L1 | 1.14 | -0.13 | 2.41 | 0.64 |
| duration | 300 prompt-control clips, 50% speech | L1 - LoRA L0 | -2.04 | -3.73 | -0.41 | 0.82 |
| duration | 500 discovery clips, 70% speech | P1 - LoRA L0 | -6.21 | -7.57 | -4.81 | 0.71 |
| duration | 500 discovery clips, 70% speech | P1 - L1 | -1.36 | -2.27 | -0.44 | 0.47 |
| duration | 500 discovery clips, 70% speech | P2 - LoRA L0 | -2.17 | -3.82 | -0.56 | 0.8 |
| duration | 500 discovery clips, 70% speech | P2 - L1 | 2.68 | 1.56 | 3.76 | 0.55 |
| duration | 500 discovery clips, 70% speech | L1 - LoRA L0 | -4.85 | -6.6 | -3.05 | 0.91 |
| snr | test batch | P1 - LoRA L0 | -4.6 | -6.03 | -3.48 | 0.67 |
| snr | test batch | P1 - L1 | 0.06 | -1.12 | 0.73 | 0.46 |
| snr | test batch | P2 - LoRA L0 | -9.34 | -10.64 | -7.87 | 0.71 |
| snr | test batch | P2 - L1 | -4.68 | -5.72 | -3.59 | 0.56 |
| snr | test batch | L1 - LoRA L0 | -4.66 | -5.77 | -3.48 | 0.59 |
| snr | 300 prompt-control clips, 50% speech | P1 - LoRA L0 | -4.88 | -6.24 | -3.53 | 0.71 |
| snr | 300 prompt-control clips, 50% speech | P1 - L1 | 0.77 | 0.21 | 1.4 | 0.31 |
| snr | 300 prompt-control clips, 50% speech | P2 - LoRA L0 | -5.09 | -6.4 | -3.86 | 0.64 |
| snr | 300 prompt-control clips, 50% speech | P2 - L1 | 0.56 | -0.72 | 1.92 | 0.65 |
| snr | 300 prompt-control clips, 50% speech | L1 - LoRA L0 | -5.65 | -7.18 | -4.19 | 0.78 |
| snr | 500 discovery clips, 70% speech | P1 - LoRA L0 | -5.23 | -6.59 | -3.85 | 0.73 |
| snr | 500 discovery clips, 70% speech | P1 - L1 | 4.95 | 3.38 | 6.56 | 0.8 |
| snr | 500 discovery clips, 70% speech | P2 - LoRA L0 | -5.25 | -6.66 | -3.86 | 0.73 |
| snr | 500 discovery clips, 70% speech | P2 - L1 | 4.94 | 3.32 | 6.55 | 0.8 |
| snr | 500 discovery clips, 70% speech | L1 - LoRA L0 | -10.18 | -12.42 | -7.92 | 1.15 |
