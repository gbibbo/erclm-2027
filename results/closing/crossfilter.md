## Heads fitted and scored on unfiltered or band-passed audio, main bank, AUROC points

| head fitted on | scored on | head AUROC | LoRA L0 AUROC | lead |
| --- | --- | --- | --- | --- |
| unfiltered | unfiltered | 95.84 | 92.67 | 3.17 |
| unfiltered | filtered | 93.58 | 90.71 | 2.88 |
| filtered | unfiltered | 94.62 | 92.67 | 1.95 |
| filtered | filtered | 95.38 | 90.71 | 4.68 | 

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| fixed heads, head AUROC change | -2.26 | -2.84 | -1.67 | 0.3 |
| LoRA L0 AUROC change | -1.97 | -2.55 | -1.37 | 0.3 |
| fixed heads, change in lead | -0.29 | -0.91 | 0.35 | 0.32 |
| refitted heads, change in lead | 1.51 | 0.89 | 2.15 | 0.32 |
