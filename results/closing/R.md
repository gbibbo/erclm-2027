## R, why H3 fails, BA and AUROC points, B = 2000

- C chosen by loso: 0.01, validation AUROC per C {0.0001: 0.847564697265625, 0.001: 0.8811442057291666, 0.01: 0.8931681315104166, 0.1: 0.8852437337239584}
- C chosen by toward_clean: 0.01, validation AUROC per C {0.0001: 0.91815185546875, 0.001: 0.93017578125, 0.01: 0.93072509765625, 0.1: 0.918212890625}
- frozen thresholds on cross-fitted training scores: {'C=0.01 (registered)': 0.35777072608470917, 'C=0.01 (leave one SNR out)': 0.35777072608470917}

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration LoRA L0 BA | 84.16 | 83.02 | 85.3 | 0.59 |
| duration C=0.01 (registered) AUROC | 87.53 | 85.94 | 88.98 | 0.78 |
| duration C=0.01 (registered) best BA | 81.33 | 80.17 | 83.13 | 0.75 |
| duration C=0.01 (registered) BA native - LoRA | -3.58 | -4.9 | -2.32 | 0.67 |
| duration C=0.01 (registered) BA batch-calibrated - LoRA | -3.23 | -4.64 | -1.89 | 0.69 |
| duration C=0.01 (registered) BA frozen threshold - LoRA | -3.79 | -5.11 | -2.51 | 0.67 |
| duration C=0.01 (leave one SNR out) AUROC | 87.53 | 85.94 | 88.98 | 0.78 |
| duration C=0.01 (leave one SNR out) best BA | 81.33 | 80.17 | 83.13 | 0.75 |
| duration C=0.01 (leave one SNR out) BA native - LoRA | -3.58 | -4.9 | -2.32 | 0.67 |
| duration C=0.01 (leave one SNR out) BA batch-calibrated - LoRA | -3.23 | -4.64 | -1.89 | 0.69 |
| duration C=0.01 (leave one SNR out) BA frozen threshold - LoRA | -3.79 | -5.11 | -2.51 | 0.67 |
| snr LoRA L0 BA | 97.37 | 96.49 | 98.15 | 0.41 |
| snr C=0.01 (registered) AUROC | 98.63 | 98.04 | 99.13 | 0.28 |
| snr C=0.01 (registered) best BA | 96.2 | 95.5 | 97.17 | 0.43 |
| snr C=0.01 (registered) BA native - LoRA | -2.34 | -3.0 | -1.68 | 0.33 |
| snr C=0.01 (registered) BA batch-calibrated - LoRA | -2.16 | -2.77 | -1.54 | 0.32 |
| snr C=0.01 (registered) BA frozen threshold - LoRA | -1.99 | -2.63 | -1.4 | 0.31 |
| snr C=0.01 (leave one SNR out) AUROC | 98.63 | 98.04 | 99.13 | 0.28 |
| snr C=0.01 (leave one SNR out) best BA | 96.2 | 95.5 | 97.17 | 0.43 |
| snr C=0.01 (leave one SNR out) BA native - LoRA | -2.34 | -3.0 | -1.68 | 0.33 |
| snr C=0.01 (leave one SNR out) BA batch-calibrated - LoRA | -2.16 | -2.77 | -1.54 | 0.32 |
| snr C=0.01 (leave one SNR out) BA frozen threshold - LoRA | -1.99 | -2.63 | -1.4 | 0.31 |
