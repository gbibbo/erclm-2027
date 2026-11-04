## Calibrating the LoRA in logit space, balanced banks, BA points, B = 2000

### main bank, 970 clips, duration axis

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| LoRA L0 | 84.16 | 83.02 | 85.3 | 0.59 |
| LoRA L1 | 85.05 | 83.98 | 86.17 | 0.56 |
| LoRA median | 86.56 | 85.16 | 87.98 | 0.71 |
| LoRA best (diagnostic) | 87.65 | 86.69 | 89.1 | 0.61 |
| recipe L1 | 83.39 | 82.04 | 84.7 | 0.67 |
| recipe median | 82.69 | 81.19 | 84.24 | 0.8 |
| LoRA median - LoRA L0 | 2.4 | 1.39 | 3.47 | 0.54 |
| LoRA best - LoRA median | 1.1 | 0.82 | 1.92 | 0.29 |
| recipe L1 - LoRA median | -3.17 | -4.22 | -2.21 | 0.52 |
| recipe median - LoRA median | -3.87 | -4.86 | -2.88 | 0.51 |
| 20 ms | recipe L1 - LoRA median | 0.41 | -2.56 | 2.93 | 1.39 |
| 20 ms | LoRA median - LoRA L0 | 7.63 | 4.62 | 10.98 | 1.64 |

### clean provenance bank, 76 recordings

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| LoRA L0 | 83.88 | 80.76 | 86.92 | 1.55 |
| LoRA L1 | 84.87 | 81.83 | 87.83 | 1.54 |
| LoRA median | 91.45 | 87.5 | 94.16 | 1.75 |
| LoRA best (diagnostic) | 92.19 | 89.72 | 95.23 | 1.45 |
| recipe L1 | 82.32 | 79.19 | 86.18 | 1.82 |
| recipe median | 84.21 | 80.67 | 88.08 | 1.89 |
| LoRA median - LoRA L0 | 7.57 | 3.95 | 10.12 | 1.59 |
| LoRA best - LoRA median | 0.74 | 0.58 | 2.96 | 0.61 |
| recipe L1 - LoRA median | -9.13 | -11.27 | -5.1 | 1.55 |
| recipe median - LoRA median | -7.24 | -9.46 | -3.7 | 1.5 |
| 20 ms | recipe L1 - LoRA median | -10.53 | -14.47 | -2.62 | 3.16 |
| 20 ms | LoRA median - LoRA L0 | 25.66 | 16.45 | 30.92 | 3.57 |

## LoRA at 20 ms on the whole main bank (970 clips), post hoc, 26 October

- 20 ms: non-speech with probability below 0.01 90.1%, median speech logit -4.58, L1 threshold -1.92, batch median -5.49, native TPR 24.7%, BA L0 61.9, BA L1 64.0, BA median 69.5, best 71.4
- 40 ms: non-speech with probability below 0.01 81.4%, median speech logit -0.78, L1 threshold -1.12, batch median -4.87, native TPR 47.0%, BA L0 72.7, BA L1 74.2, BA median 78.1, best 79.6
