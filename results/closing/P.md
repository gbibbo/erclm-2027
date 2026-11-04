## P, calibration that should not depend on the speech share, on the average of ('ab', 'ba_speech_second'), BA points, B = 2000

{'held-out clips': 470, 'P1 failed fits in the sweep': 2, 'Platt map (slope, intercept)': (1.347544706787528, 2.3265916072077695), 'training speech share': 0.5, 'fixed-batch speech share estimates': {'300 prompt-control clips, 50% speech': {'P1': 0.4384488851597762, 'P2': 0.9999990000000002}, '500 discovery clips, 70% speech': {'P1': 0.5895452298168686, 'P2': 0.9999990000000002}}} 

### Reading of the sweep

|  | range | lowest | L1 at 50% | flat (range <= 2) | no cost (lowest >= L1 at 50% - 2) |
| --- | --- | --- | --- | --- | --- |
| duration | L1 | 8.704895660585976 | 76.00368297401879 | 84.6068891652847 | False | False |
| duration | P1 | 1.0763059701492637 | 82.66061705362077 | 84.6068891652847 | True | True |
| duration | P1 own variances | 2.746814538419031 | 81.05067371475953 | 84.6068891652847 | False | False |
| duration | P2 | 5.984207435046983 | 78.62287175234937 | 84.6068891652847 | False | False |
| snr | L1 | 12.899631472268283 | 78.81151649161599 | 90.14738805970148 | False | False |
| snr | P1 | 0.3758982863460574 | 90.67460843928504 | 90.14738805970148 | True | True |
| snr | P1 own variances | 2.79131656532158 | 87.9322093237516 | 90.14738805970148 | False | False |
| snr | P2 | 6.533890731527563 | 84.46683250414593 | 90.14738805970148 | False | False |

### Sweep, 150 calibration clips from the discovery recordings, the 200 draws of g1t

| axis | method | speech share | BA | 2.5% | 97.5% | BA - LoRA L0 | estimated share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| duration | L1 | 0.1 | 81.17 | 80.07 | 82.14 | -3.38 | nan |
| duration | P1 | 0.1 | 83.74 | 83.04 | 84.43 | -0.81 | 0.16 |
| duration | P1 own variances | 0.1 | 82.99 | 81.49 | 84.22 | -1.56 | nan |
| duration | P2 | 0.1 | 78.62 | 77.88 | 79.2 | -5.93 | 0.71 |
| snr | L1 | 0.1 | 88.65 | 87.8 | 89.3 | -9.35 | nan |
| snr | P1 | 0.1 | 90.67 | 90.01 | 91.12 | -7.33 | 0.16 |
| snr | P1 own variances | 0.1 | 87.93 | 86.86 | 88.73 | -10.07 | nan |
| snr | P2 | 0.1 | 88.97 | 88.57 | 89.4 | -9.03 | 0.71 |
| duration | L1 | 0.2 | 83.77 | 83.16 | 84.31 | -0.78 | nan |
| duration | P1 | 0.2 | 83.62 | 82.99 | 84.16 | -0.93 | 0.22 |
| duration | P1 own variances | 0.2 | 83.6 | 82.39 | 84.66 | -0.95 | nan |
| duration | P2 | 0.2 | 80.48 | 79.84 | 81.5 | -4.07 | 0.83 |
| snr | L1 | 0.2 | 90.65 | 90.41 | 90.96 | -7.35 | nan |
| snr | P1 | 0.2 | 90.94 | 90.31 | 91.33 | -7.06 | 0.22 |
| snr | P1 own variances | 0.2 | 88.78 | 87.68 | 90.63 | -9.22 | nan |
| snr | P2 | 0.2 | 89.48 | 89.22 | 89.83 | -8.52 | 0.83 |
| duration | L1 | 0.3 | 84.71 | 84.26 | 85.08 | 0.16 | nan |
| duration | P1 | 0.3 | 83.57 | 83.01 | 84.12 | -0.99 | 0.29 |
| duration | P1 own variances | 0.3 | 83.8 | 82.47 | 84.68 | -0.75 | nan |
| duration | P2 | 0.3 | 82.51 | 81.45 | 83.34 | -2.04 | 0.92 |
| snr | L1 | 0.3 | 91.71 | 91.41 | 91.99 | -6.29 | nan |
| snr | P1 | 0.3 | 91.05 | 90.5 | 91.47 | -6.95 | 0.29 |
| snr | P1 own variances | 0.3 | 89.58 | 87.53 | 91.42 | -8.42 | nan |
| snr | P2 | 0.3 | 89.81 | 89.47 | 90.18 | -8.19 | 0.92 |
| duration | L1 | 0.5 | 84.61 | 83.97 | 85.12 | 0.06 | nan |
| duration | P1 | 0.5 | 83.33 | 82.54 | 83.97 | -1.22 | 0.44 |
| duration | P1 own variances | 0.5 | 83.36 | 81.56 | 84.53 | -1.19 | nan |
| duration | P2 | 0.5 | 84.57 | 84.17 | 85.04 | 0.02 | 1.0 |
| snr | L1 | 0.5 | 90.15 | 89.39 | 91.03 | -7.85 | nan |
| snr | P1 | 0.5 | 91.03 | 90.45 | 91.47 | -6.97 | 0.44 |
| snr | P1 own variances | 0.5 | 90.72 | 88.71 | 91.69 | -7.28 | nan |
| snr | P2 | 0.5 | 91.0 | 90.61 | 91.42 | -7.0 | 1.0 |
| duration | L1 | 0.7 | 81.11 | 79.56 | 82.26 | -3.44 | nan |
| duration | P1 | 0.7 | 83.19 | 82.4 | 83.76 | -1.36 | 0.59 |
| duration | P1 own variances | 0.7 | 83.0 | 80.5 | 84.35 | -1.55 | nan |
| duration | P2 | 0.7 | 84.61 | 83.65 | 85.13 | 0.06 | 1.0 |
| snr | L1 | 0.7 | 85.84 | 84.67 | 86.77 | -12.16 | nan |
| snr | P1 | 0.7 | 90.94 | 90.08 | 91.52 | -7.06 | 0.59 |
| snr | P1 own variances | 0.7 | 90.7 | 88.6 | 91.85 | -7.3 | nan |
| snr | P2 | 0.7 | 90.93 | 90.07 | 91.75 | -7.07 | 1.0 |
| duration | L1 | 0.9 | 76.0 | 74.92 | 77.28 | -8.55 | nan |
| duration | P1 | 0.9 | 82.66 | 80.79 | 83.52 | -1.89 | 0.75 |
| duration | P1 own variances | 0.9 | 81.05 | 78.92 | 83.27 | -3.5 | nan |
| duration | P2 | 0.9 | 80.15 | 78.68 | 81.61 | -4.41 | 1.0 |
| snr | L1 | 0.9 | 78.81 | 77.22 | 80.45 | -19.19 | nan |
| snr | P1 | 0.9 | 90.68 | 89.5 | 91.59 | -7.32 | 0.75 |
| snr | P1 own variances | 0.9 | 88.51 | 85.0 | 91.15 | -9.49 | nan |
| snr | P2 | 0.9 | 84.47 | 82.66 | 86.08 | -13.53 | 1.0 |

### Paired bootstrap on the 470 held-out clips

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration | test batch | P1 - LoRA L0 | -0.85 | -2.54 | 0.53 | 0.79 |
| duration | test batch | P1 - L1 | -1.11 | -2.52 | 0.22 | 0.68 |
| duration | test batch | P2 - LoRA L0 | -2.18 | -4.09 | -0.29 | 0.96 |
| duration | test batch | P2 - L1 | -2.45 | -3.42 | -1.27 | 0.55 |
| duration | test batch | L1 - LoRA L0 | 0.26 | -1.73 | 1.99 | 0.96 |
| duration | 300 prompt-control clips, 50% speech | P1 - LoRA L0 | -1.19 | -2.62 | 0.38 | 0.76 |
| duration | 300 prompt-control clips, 50% speech | P1 - L1 | -1.37 | -2.24 | -0.49 | 0.44 |
| duration | 300 prompt-control clips, 50% speech | P2 - LoRA L0 | -0.21 | -2.07 | 1.7 | 0.99 |
| duration | 300 prompt-control clips, 50% speech | P2 - L1 | -0.38 | -1.31 | 0.55 | 0.48 |
| duration | 300 prompt-control clips, 50% speech | L1 - LoRA L0 | 0.17 | -1.43 | 1.9 | 0.85 |
| duration | 500 discovery clips, 70% speech | P1 - LoRA L0 | -1.26 | -2.73 | 0.28 | 0.76 |
| duration | 500 discovery clips, 70% speech | P1 - L1 | 1.85 | 0.79 | 2.91 | 0.53 |
| duration | 500 discovery clips, 70% speech | P2 - LoRA L0 | 0.31 | -1.31 | 2.06 | 0.86 |
| duration | 500 discovery clips, 70% speech | P2 - L1 | 3.41 | 2.23 | 4.58 | 0.59 |
| duration | 500 discovery clips, 70% speech | L1 - LoRA L0 | -3.1 | -4.84 | -1.35 | 0.89 |
| snr | test batch | P1 - LoRA L0 | -7.51 | -9.04 | -5.76 | 0.84 |
| snr | test batch | P1 - L1 | -1.11 | -1.76 | -0.27 | 0.39 |
| snr | test batch | P2 - LoRA L0 | -7.99 | -9.6 | -6.46 | 0.8 |
| snr | test batch | P2 - L1 | -1.6 | -2.53 | -0.74 | 0.46 |
| snr | test batch | L1 - LoRA L0 | -6.39 | -7.68 | -5.08 | 0.68 |
| snr | 300 prompt-control clips, 50% speech | P1 - LoRA L0 | -6.81 | -8.34 | -5.28 | 0.79 |
| snr | 300 prompt-control clips, 50% speech | P1 - L1 | 1.05 | -0.09 | 2.29 | 0.61 |
| snr | 300 prompt-control clips, 50% speech | P2 - LoRA L0 | -6.99 | -8.4 | -5.6 | 0.72 |
| snr | 300 prompt-control clips, 50% speech | P2 - L1 | 0.86 | -0.56 | 2.33 | 0.74 |
| snr | 300 prompt-control clips, 50% speech | L1 - LoRA L0 | -7.86 | -9.76 | -6.09 | 0.96 |
| snr | 500 discovery clips, 70% speech | P1 - LoRA L0 | -6.9 | -8.45 | -5.34 | 0.8 |
| snr | 500 discovery clips, 70% speech | P1 - L1 | 5.31 | 3.23 | 7.44 | 1.06 |
| snr | 500 discovery clips, 70% speech | P2 - LoRA L0 | -6.95 | -8.75 | -5.22 | 0.93 |
| snr | 500 discovery clips, 70% speech | P2 - L1 | 5.26 | 3.6 | 6.86 | 0.81 |
| snr | 500 discovery clips, 70% speech | L1 - LoRA L0 | -12.2 | -14.78 | -9.61 | 1.33 |
