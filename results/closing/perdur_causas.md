## Per condition on the 470 held-out clips, post hoc diagnostics (26 October)

| condition | recipe AUROC | LoRA AUROC | recipe L1 | recipe best | LoRA L0 | LoRA L1 | LoRA best | LoRA L0 TPR | LoRA L0 TNR | recipe L1 TPR | recipe L1 TNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dur_20ms | 75.2 | 80.2 | 68.4 | 70.6 | 60.3 | 64.1 | 73.1 | 21.5 | 99.1 | 63.7 | 73.1 |
| dur_40ms | 84.7 | 88.0 | 75.6 | 77.7 | 71.6 | 74.1 | 80.1 | 44.4 | 98.8 | 71.1 | 80.0 |
| dur_60ms | 92.6 | 93.3 | 85.4 | 86.5 | 83.4 | 84.4 | 85.1 | 68.9 | 97.9 | 81.5 | 89.3 |
| dur_80ms | 93.6 | 95.3 | 85.6 | 86.9 | 83.9 | 84.9 | 89.4 | 68.1 | 99.7 | 85.2 | 86.0 |
| dur_100ms | 94.7 | 96.5 | 87.7 | 88.9 | 86.5 | 88.7 | 91.2 | 74.8 | 98.2 | 88.9 | 86.6 |
| dur_200ms | 96.6 | 99.1 | 90.8 | 91.8 | 93.1 | 94.4 | 94.8 | 87.4 | 98.8 | 94.1 | 87.5 |
| dur_500ms | 97.0 | 99.8 | 92.9 | 93.2 | 98.4 | 98.7 | 98.8 | 97.8 | 99.1 | 96.3 | 89.6 |
| dur_1000ms | 96.8 | 100.0 | 92.2 | 92.2 | 99.1 | 99.2 | 99.6 | 98.5 | 99.7 | 94.8 | 89.6 |
| snr_-10dB | 96.9 | 99.1 | 91.0 | 94.3 | 95.8 | 95.1 | 97.4 | 98.5 | 93.1 | 100.0 | 82.1 |
| snr_-5dB | 96.8 | 99.8 | 90.1 | 92.5 | 97.9 | 97.6 | 98.5 | 100.0 | 95.8 | 97.8 | 82.4 |
| snr_0dB | 95.5 | 99.8 | 91.6 | 92.6 | 98.1 | 97.8 | 98.9 | 99.3 | 97.0 | 97.8 | 85.4 |
| snr_5dB | 96.8 | 99.9 | 93.8 | 94.0 | 98.6 | 98.1 | 99.0 | 99.3 | 97.9 | 97.8 | 89.9 |
| snr_10dB | 96.7 | 100.0 | 92.0 | 92.2 | 98.8 | 98.7 | 99.6 | 100.0 | 97.6 | 94.1 | 89.9 |
| snr_20dB | 95.3 | 100.0 | 91.1 | 91.1 | 98.7 | 99.0 | 99.2 | 99.3 | 98.2 | 91.9 | 90.4 |

## LoRA native decision by duration, whole test bank against its own training bank

| bank | duration ms | SNR dB | TPR at 0 | TNR at 0 | BA at 0 | best BA |
| --- | --- | --- | --- | --- | --- | --- |
| test, all 970 clips, clean container | 20 | none | 24.7 | 99.0 | 61.9 | 71.4 |
| test, all 970 clips, clean container | 40 | none | 47.0 | 98.4 | 72.7 | 79.6 |
| test, all 970 clips, clean container | 100 | none | 74.4 | 97.1 | 85.8 | 89.3 |
| test, all 970 clips, clean container | 1000 | none | 96.5 | 99.6 | 98.0 | 98.6 |
| LoRA training bank, noisy container | 20 | -10 | 34.4 | 87.5 | 60.9 | 70.3 |
| LoRA training bank, noisy container | 20 | 0 | 68.8 | 96.9 | 82.8 | 85.9 |
| LoRA training bank, noisy container | 20 | 20 | 68.8 | 96.9 | 82.8 | 87.5 |
| LoRA training bank, noisy container | 40 | -10 | 56.2 | 81.2 | 68.8 | 76.6 |
| LoRA training bank, noisy container | 40 | 0 | 78.1 | 96.9 | 87.5 | 92.2 |
| LoRA training bank, noisy container | 40 | 20 | 75.0 | 100.0 | 87.5 | 89.1 |
| LoRA training bank, noisy container | 100 | -10 | 81.2 | 93.8 | 87.5 | 89.1 |
| LoRA training bank, noisy container | 100 | 0 | 93.8 | 100.0 | 96.9 | 100.0 |
| LoRA training bank, noisy container | 100 | 20 | 96.9 | 100.0 | 98.4 | 100.0 |
| LoRA training bank, noisy container | 1000 | -10 | 96.9 | 96.9 | 96.9 | 98.4 |
| LoRA training bank, noisy container | 1000 | 0 | 100.0 | 100.0 | 100.0 | 100.0 |
| LoRA training bank, noisy container | 1000 | 20 | 100.0 | 100.0 | 100.0 | 100.0 |
