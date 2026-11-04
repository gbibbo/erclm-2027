## Provenance results without doubtful negatives, AUROC points, B = 2000

| negatives dropped | quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- | --- |
| none | LoRA L0, provenance | 92.95 | 90.32 | 95.23 | 1.29 |
| none | LoRA L0 at 1000 ms, provenance | 96.4 | 93.71 | 98.37 | 1.2 |
| none | in-bank lead, provenance | 0.14 | -1.11 | 1.48 | 0.66 |
| none | B gap of the same heads | 4.69 | 1.9 | 7.73 | 1.5 |
| p >= 0.5 (18 recordings) | LoRA L0, provenance | 94.7 | 92.07 | 96.87 | 1.23 |
| p >= 0.5 (18 recordings) | LoRA L0 at 1000 ms, provenance | 98.86 | 97.69 | 99.67 | 0.54 |
| p >= 0.5 (18 recordings) | in-bank lead, provenance | -0.45 | -1.64 | 0.8 | 0.62 |
| p >= 0.5 (18 recordings) | B gap of the same heads | 2.63 | 0.15 | 5.76 | 1.41 |
| p >= 0.3 (27 recordings) | LoRA L0, provenance | 94.79 | 92.01 | 96.98 | 1.26 |
| p >= 0.3 (27 recordings) | LoRA L0 at 1000 ms, provenance | 99.39 | 98.48 | 99.93 | 0.37 |
| p >= 0.3 (27 recordings) | in-bank lead, provenance | -0.3 | -1.54 | 1.05 | 0.64 |
| p >= 0.3 (27 recordings) | B gap of the same heads | 2.3 | -0.15 | 5.06 | 1.36 | 

