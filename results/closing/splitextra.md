## Restricted LoRA, native and best-threshold BA, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| duration base BA | 50.08 | 50.0 | 50.21 | 0.06 |
| duration base AUROC | 86.65 | 83.84 | 89.2 | 1.37 |
| duration base best-threshold BA | 81.08 | 79.33 | 84.04 | 1.22 |
| duration lora BA | 83.71 | 81.46 | 85.88 | 1.13 |
| duration lora AUROC | 91.9 | 89.4 | 94.11 | 1.2 |
| duration lora best-threshold BA | 87.5 | 85.87 | 90.33 | 1.13 |
| duration split-audio BA | 50.08 | 50.0 | 50.21 | 0.06 |
| duration split-audio AUROC | 89.83 | 87.24 | 92.07 | 1.23 |
| duration split-audio best-threshold BA | 84.75 | 82.92 | 87.42 | 1.17 |
| duration split-llm BA | 82.21 | 79.96 | 84.38 | 1.12 |
| duration split-llm AUROC | 90.54 | 87.89 | 92.92 | 1.28 |
| duration split-llm best-threshold BA | 86.08 | 84.17 | 88.88 | 1.18 |
| duration split-llm-lower BA | 73.25 | 71.33 | 75.08 | 0.95 |
| duration split-llm-lower AUROC | 90.24 | 87.52 | 92.68 | 1.31 |
| duration split-llm-lower best-threshold BA | 86.04 | 84.04 | 88.75 | 1.19 |
| duration split-llm-upper BA | 51.21 | 50.79 | 51.71 | 0.24 |
| duration split-llm-upper AUROC | 89.9 | 87.25 | 92.25 | 1.25 |
| duration split-llm-upper best-threshold BA | 85.33 | 83.42 | 88.04 | 1.19 |
| duration encoder effect given base LM, AUROC | 3.19 | 2.36 | 4.06 | 0.43 |
| duration encoder effect given adapted LM, AUROC | 1.36 | 0.84 | 1.91 | 0.28 |
| duration encoder effect given base LM, BA | 0.0 | 0.0 | 0.0 | 0.0 |
| duration encoder effect given adapted LM, BA | 1.5 | 0.88 | 2.12 | 0.32 |
| duration encoder effect given base LM, best-threshold BA | 3.67 | 2.54 | 4.42 | 0.48 |
| duration encoder effect given adapted LM, best-threshold BA | 1.42 | 0.79 | 2.46 | 0.44 |
| snr base BA | 50.06 | 50.0 | 50.17 | 0.05 |
| snr base AUROC | 91.96 | 89.2 | 94.39 | 1.32 |
| snr base best-threshold BA | 86.22 | 84.06 | 89.39 | 1.36 |
| snr lora BA | 96.72 | 94.83 | 98.22 | 0.87 |
| snr lora AUROC | 98.85 | 97.64 | 99.69 | 0.52 |
| snr lora best-threshold BA | 97.5 | 96.17 | 99.0 | 0.72 |
| snr split-audio BA | 50.44 | 50.0 | 51.11 | 0.28 |
| snr split-audio AUROC | 94.48 | 91.71 | 96.73 | 1.29 |
| snr split-audio best-threshold BA | 91.56 | 89.56 | 94.11 | 1.15 |
| snr split-llm BA | 97.17 | 95.5 | 98.56 | 0.78 |
| snr split-llm AUROC | 99.23 | 98.39 | 99.85 | 0.37 |
| snr split-llm best-threshold BA | 97.89 | 96.56 | 99.17 | 0.67 |
| snr split-llm-lower BA | 95.72 | 93.94 | 97.22 | 0.83 |
| snr split-llm-lower AUROC | 99.02 | 98.0 | 99.75 | 0.45 |
| snr split-llm-lower best-threshold BA | 97.33 | 96.0 | 98.83 | 0.73 |
| snr split-llm-upper BA | 52.5 | 51.44 | 53.72 | 0.6 |
| snr split-llm-upper AUROC | 97.39 | 95.53 | 98.84 | 0.83 |
| snr split-llm-upper best-threshold BA | 94.83 | 93.06 | 96.89 | 0.98 |
| snr encoder effect given base LM, AUROC | 2.52 | 1.08 | 3.95 | 0.74 |
| snr encoder effect given adapted LM, AUROC | -0.39 | -0.89 | 0.01 | 0.23 |
| snr encoder effect given base LM, BA | 0.39 | 0.0 | 0.94 | 0.24 |
| snr encoder effect given adapted LM, BA | -0.44 | -1.17 | 0.17 | 0.34 |
| snr encoder effect given base LM, best-threshold BA | 5.33 | 3.55 | 6.83 | 0.86 |
| snr encoder effect given adapted LM, best-threshold BA | -0.39 | -1.0 | 0.39 | 0.37 |
