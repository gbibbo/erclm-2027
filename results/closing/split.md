## I2, LoRA kept in one part of the model, B = 2000

|  | mean over the axis |
| --- | --- |
| duration base auroc | 86.6 |
| duration base ba | 50.1 |
| duration lora auroc | 91.9 |
| duration lora ba | 83.7 |
| duration split-audio auroc | 89.8 |
| duration split-audio ba | 50.1 |
| duration split-llm auroc | 90.5 |
| duration split-llm ba | 82.2 |
| duration split-llm-lower auroc | 90.2 |
| duration split-llm-lower ba | 73.2 |
| duration split-llm-upper auroc | 89.9 |
| duration split-llm-upper ba | 51.2 |
| snr base auroc | 92.0 |
| snr base ba | 50.1 |
| snr lora auroc | 98.8 |
| snr lora ba | 96.7 |
| snr split-audio auroc | 94.5 |
| snr split-audio ba | 50.4 |
| snr split-llm auroc | 99.2 |
| snr split-llm ba | 97.2 |
| snr split-llm-lower auroc | 99.0 |
| snr split-llm-lower ba | 95.7 |
| snr split-llm-upper auroc | 97.4 |
| snr split-llm-upper ba | 52.5 | 

| quantity | estimate | lo | hi |
| --- | --- | --- | --- |
| duration R split-audio | 0.607 | 0.496 | 0.738 |
| duration R split-llm | 0.741 | 0.623 | 0.835 |
| duration R split-llm-lower | 0.684 | 0.565 | 0.782 |
| duration R split-llm-upper | 0.62 | 0.54 | 0.703 |
| duration auroc LoRA minus base | 0.053 | 0.039 | 0.066 |
| duration C split-audio | 0.0 | 0.0 | 0.0 |
| duration C split-llm | 0.955 | 0.937 | 0.974 |
| duration C split-llm-lower | 0.689 | 0.652 | 0.728 |
| duration C split-llm-upper | 0.033 | 0.021 | 0.047 |
| duration ba LoRA minus base | 0.336 | 0.314 | 0.358 |
| snr R split-audio | 0.366 | 0.176 | 0.538 |
| snr R split-llm | 1.056 | 0.999 | 1.141 |
| snr R split-llm-lower | 1.024 | 0.975 | 1.098 |
| snr R split-llm-upper | 0.789 | 0.687 | 0.907 |
| snr auroc LoRA minus base | 0.069 | 0.049 | 0.092 |
| snr C split-audio | 0.008 | 0.0 | 0.02 |
| snr C split-llm | 1.01 | 0.996 | 1.025 |
| snr C split-llm-lower | 0.979 | 0.954 | 1.005 |
| snr C split-llm-upper | 0.052 | 0.03 | 0.078 |
| snr ba LoRA minus base | 0.467 | 0.448 | 0.482 | 

### Reading rule

| axis | share | pair | reading |
| --- | --- | --- | --- |
| duration | R | split-audio / split-llm | mixed |
| duration | R | split-llm-lower / split-llm-upper | either part suffices |
| duration | C | split-audio / split-llm | split-llm carries it |
| duration | C | split-llm-lower / split-llm-upper | split-llm-lower carries it |
| snr | R | split-audio / split-llm | mixed |
| snr | R | split-llm-lower / split-llm-upper | either part suffices |
| snr | C | split-audio / split-llm | split-llm carries it |
| snr | C | split-llm-lower / split-llm-upper | split-llm-lower carries it | 

