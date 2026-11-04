## 20 and 40 ms, main bank: speech against each ESC-50 major category (post hoc follow-up of N)

| negatives | n negatives | LoRA L0 AUROC | head 122 AUROC | head full AUROC | base native AUROC | head 122 - LoRA |
| --- | --- | --- | --- | --- | --- | --- |
| animals | 170.0 | 83.0 | 83.2 | 85.0 | 76.8 | 0.2 |
| natural soundscapes | 202.0 | 86.6 | 91.6 | 93.1 | 78.1 | 5.0 |
| human non-speech | 214.0 | 77.0 | 85.4 | 89.7 | 72.2 | 8.4 |
| interior/domestic | 222.0 | 78.5 | 90.0 | 92.4 | 69.5 | 11.5 |
| exterior/urban | 162.0 | 92.1 | 92.6 | 93.5 | 86.5 | 0.6 |

Ten ESC-50 classes with the lowest LoRA AUROC at 20 and 40 ms

| esc class | n | LoRA L0 AUROC | head 122 AUROC | head - LoRA |
| --- | --- | --- | --- | --- |
| 21 | 28.0 | 61.4 | 77.8 | 16.5 |
| 24 | 24.0 | 62.2 | 77.9 | 15.7 |
| 7 | 14.0 | 67.1 | 67.9 | 0.8 |
| 34 | 26.0 | 68.9 | 87.3 | 18.4 |
| 15 | 22.0 | 69.8 | 89.7 | 19.9 |
| 39 | 24.0 | 70.2 | 88.9 | 18.7 |
| 31 | 22.0 | 71.0 | 86.5 | 15.5 |
| 0 | 28.0 | 71.3 | 72.9 | 1.6 |
| 29 | 18.0 | 71.5 | 89.1 | 17.6 |
| 14 | 18.0 | 73.3 | 79.8 | 6.5 |
