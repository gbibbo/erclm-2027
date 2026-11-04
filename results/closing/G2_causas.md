## G2 causes, post hoc, AUROC per condition on the 470 held-out clips (25 October)

| condition | ab (A speech, first) | ba_speech_second (B speech, second) | ab_speech_second (A speech, second) | s_sim | s_seg |
| --- | --- | --- | --- | --- | --- |
| dur_20ms | 71.6 | 74.7 | 60.7 | 75.2 | 74.7 |
| dur_40ms | 82.2 | 83.5 | 70.8 | 84.7 | 82.4 |
| dur_60ms | 90.3 | 92.1 | 81.1 | 92.6 | 91.4 |
| dur_80ms | 91.4 | 92.0 | 83.0 | 93.6 | 92.0 |
| dur_100ms | 93.5 | 93.2 | 87.3 | 94.7 | 93.8 |
| dur_200ms | 95.2 | 94.1 | 90.4 | 96.6 | 96.7 |
| dur_500ms | 95.5 | 91.9 | 95.7 | 97.0 | 97.2 |
| dur_1000ms | 95.2 | 91.7 | 97.7 | 96.8 | 97.5 |
| snr_-10dB | 92.7 | 97.5 | 97.4 | 96.9 | 98.1 |
| snr_-5dB | 91.2 | 97.6 | 97.5 | 96.8 | 98.4 |
| snr_0dB | 92.0 | 95.6 | 97.6 | 95.5 | 97.6 |
| snr_5dB | 93.4 | 96.0 | 97.2 | 96.8 | 98.2 |
| snr_10dB | 93.6 | 95.4 | 96.6 | 96.7 | 97.9 |
| snr_20dB | 92.1 | 92.5 | 94.8 | 95.3 | 96.5 |

- axis means:
  - dur: ab (A speech, first) 89.38, ba_speech_second (B speech, second) 89.14, ab_speech_second (A speech, second) 83.36, s_sim 91.41, s_seg 90.70
  - snr: ab (A speech, first) 92.51, ba_speech_second (B speech, second) 95.76, ab_speech_second (A speech, second) 96.86, s_sim 96.33, s_seg 97.78

### Within-class correlation of the two formats in each average, mean over conditions

- s_sim dur: 0.46
- s_sim snr: 0.45
- s_seg dur: 0.35
- s_seg snr: 0.46

## Discovery only. The three informative formats on the 300 prompt-control clips, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration | s_sim | AUROC | 88.77 | 85.96 | 91.31 | 1.35 |
| duration | s_sim | BA L1 - LoRA L0 | -0.96 | -2.58 | 0.83 | 0.87 |
| duration | s_seg | AUROC | 88.37 | 85.57 | 90.86 | 1.35 |
| duration | s_seg | BA L1 - LoRA L0 | -1.62 | -3.12 | 0.04 | 0.8 |
| duration | s_3 (ab, ba_speech_second, ab_speech_second) | AUROC | 88.59 | 85.83 | 91.09 | 1.33 |
| duration | s_3 (ab, ba_speech_second, ab_speech_second) | BA L1 - LoRA L0 | -2.08 | -3.5 | -0.79 | 0.72 |
| snr | s_sim | AUROC | 96.17 | 94.41 | 97.64 | 0.85 |
| snr | s_sim | BA L1 - LoRA L0 | -7.0 | -8.94 | -5.06 | 0.98 |
| snr | s_seg | AUROC | 97.73 | 96.46 | 98.7 | 0.58 |
| snr | s_seg | BA L1 - LoRA L0 | -4.72 | -6.44 | -3.06 | 0.89 |
| snr | s_3 (ab, ba_speech_second, ab_speech_second) | AUROC | 97.0 | 95.5 | 98.21 | 0.71 |
| snr | s_3 (ab, ba_speech_second, ab_speech_second) | BA L1 - LoRA L0 | -6.89 | -8.61 | -5.06 | 0.91 |
