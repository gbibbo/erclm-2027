## G1 transfer, thresholds from separate unlabeled clips, BA points, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration recipe on its own batch - LoRA L0 | 0.26 | -1.73 | 1.99 | 0.96 |
| duration recipe, thresholds from prompt-control clips (300, 50% speech) - LoRA L0 | 0.17 | -1.43 | 1.9 | 0.85 |
| duration recipe, thresholds from all clips of the discovery recordings (500, 70% speech) - LoRA L0 | -3.1 | -4.84 | -1.35 | 0.89 |
| snr recipe on its own batch - LoRA L0 | -6.39 | -7.68 | -5.08 | 0.68 |
| snr recipe, thresholds from prompt-control clips (300, 50% speech) - LoRA L0 | -7.86 | -9.76 | -6.09 | 0.96 |
| snr recipe, thresholds from all clips of the discovery recordings (500, 70% speech) - LoRA L0 | -12.2 | -14.78 | -9.61 | 1.33 |

Speech share of a calibration batch of 150 clips from the discovery recordings, 200 draws

| axis | speech share of the calibration batch | recipe BA | 2.5% | 97.5% | recipe - LoRA L0 |
| --- | --- | --- | --- | --- | --- |
| duration | 0.1 | 81.17 | 80.07 | 82.14 | -3.38 |
| snr | 0.1 | 88.65 | 87.8 | 89.3 | -9.35 |
| duration | 0.2 | 83.77 | 83.16 | 84.31 | -0.78 |
| snr | 0.2 | 90.65 | 90.41 | 90.96 | -7.35 |
| duration | 0.3 | 84.71 | 84.26 | 85.08 | 0.16 |
| snr | 0.3 | 91.71 | 91.41 | 91.99 | -6.29 |
| duration | 0.5 | 84.61 | 83.97 | 85.12 | 0.06 |
| snr | 0.5 | 90.15 | 89.39 | 91.03 | -7.85 |
| duration | 0.7 | 81.11 | 79.56 | 82.26 | -3.44 |
| snr | 0.7 | 85.84 | 84.67 | 86.77 | -12.16 |
| duration | 0.9 | 76.0 | 74.92 | 77.28 | -8.55 |
| snr | 0.9 | 78.81 | 77.22 | 80.45 | -19.19 |

## Does the average beat both formats? AUROC points on the 470 held-out clips, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration AUROC original (ab) | 89.38 | 87.42 | 91.19 | 0.97 |
| duration AUROC ba_speech_second | 89.14 | 86.91 | 91.2 | 1.07 |
| duration AUROC average | 91.41 | 89.55 | 93.14 | 0.93 |
| duration AUROC average of z-scores | 91.44 | 89.57 | 93.16 | 0.92 |
| duration average - better single format | 2.03 | 1.03 | 2.32 | 0.33 |
| snr AUROC original (ab) | 92.51 | 90.43 | 94.28 | 0.99 |
| snr AUROC ba_speech_second | 95.76 | 94.48 | 96.98 | 0.63 |
| snr AUROC average | 96.33 | 95.04 | 97.42 | 0.59 |
| snr AUROC average of z-scores | 96.2 | 94.88 | 97.32 | 0.61 |
| snr average - better single format | 0.57 | -0.06 | 1.23 | 0.33 |

| axis | class | mean within-condition correlation of the two formats |
| --- | --- | --- |
| duration | non-speech | 0.37 |
| duration | speech | 0.56 |
| snr | non-speech | 0.52 |
| snr | speech | 0.37 |

## AUROC of each prompt format alone on the 300 prompt-control clips, base, B = 2000
ab = A) speech B) non-speech, speech listed first. ab_speech_second = A) speech, listed second.
ba_speech_first = A) non-speech, speech listed first. ba_speech_second = A) non-speech B) speech.

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration AUROC ab | 86.65 | 83.84 | 89.2 | 1.37 |
| duration AUROC ab_speech_second | 82.84 | 79.94 | 85.64 | 1.46 |
| duration AUROC ba_speech_first | 39.68 | 35.53 | 43.7 | 2.07 |
| duration AUROC ba_speech_second | 86.68 | 83.81 | 89.44 | 1.46 |
| snr AUROC ab | 91.96 | 89.2 | 94.39 | 1.32 |
| snr AUROC ab_speech_second | 96.41 | 94.82 | 97.71 | 0.75 |
| snr AUROC ba_speech_first | 29.71 | 25.12 | 34.03 | 2.23 |
| snr AUROC ba_speech_second | 96.09 | 94.36 | 97.49 | 0.8 |

## Discovery only. Averaging the two speech-second formats, 300 prompt-control clips, base, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration AUROC sym2 (ab + ba_speech_second) | 88.77 | 85.96 | 91.31 | 1.35 |
| duration BA L1 sym2 (ab + ba_speech_second) - LoRA L0 | -0.96 | -2.58 | 0.83 | 0.87 |
| duration AUROC second2 (ab_speech_second + ba_speech_second) | 88.37 | 85.57 | 90.86 | 1.35 |
| duration BA L1 second2 (ab_speech_second + ba_speech_second) - LoRA L0 | -1.62 | -3.12 | 0.04 | 0.8 |
| duration AUROC second2 - sym2 | -0.4 | -1.11 | 0.26 | 0.35 |
| snr AUROC sym2 (ab + ba_speech_second) | 96.17 | 94.41 | 97.64 | 0.85 |
| snr BA L1 sym2 (ab + ba_speech_second) - LoRA L0 | -7.0 | -8.94 | -5.06 | 0.98 |
| snr AUROC second2 (ab_speech_second + ba_speech_second) | 97.73 | 96.46 | 98.7 | 0.58 |
| snr BA L1 second2 (ab_speech_second + ba_speech_second) - LoRA L0 | -4.72 | -6.44 | -3.06 | 0.89 |
| snr AUROC second2 - sym2 | 1.56 | 0.88 | 2.34 | 0.38 |
