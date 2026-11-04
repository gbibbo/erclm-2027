fold heads f in 2.0 min, C per fold [0.01, 0.01, 0.01, 0.01, 0.01]
## I1, channel-cue intervention, AUROC points, B = 2000

| quantity | estimate | lo | hi | se |
| --- | --- | --- | --- | --- |
| A lead, unfiltered | 3.17 | 2.35 | 4.08 | 0.45 |
| A lead, filtered | 4.68 | 3.69 | 5.7 | 0.52 |
| A change | 1.51 | 0.89 | 2.09 | 0.31 |
| B gap, unfiltered | 4.69 | 1.9 | 7.73 | 1.5 |
| B gap, filtered | 4.43 | 1.56 | 7.63 | 1.54 |
| B change | -0.26 | -1.32 | 0.88 | 0.58 |
| B gap, unfiltered, heard-clean negatives | 2.32 | -0.35 | 5.48 | 1.49 |
| B gap, filtered, heard-clean negatives | 2.26 | -0.48 | 5.56 | 1.51 |
| B change, heard-clean negatives | -0.06 | -1.22 | 1.16 | 0.61 |
| control main s_lora: filtered - unfiltered | -1.97 | -2.54 | -1.34 | 0.31 |
| control main s_base: filtered - unfiltered | -3.3 | -4.3 | -2.31 | 0.51 |
| control provenance s_lora: filtered - unfiltered | -1.93 | -3.11 | -0.73 | 0.62 |
| control provenance s_base: filtered - unfiltered | -2.88 | -4.52 | -1.3 | 0.84 | 

### Reading rule

| estimand | original | estimate | lo | hi | reading |
| --- | --- | --- | --- | --- | --- |
| A change | 3.17 | 1.51 | 0.89 | 2.09 | the filter does not explain it |
| B change | 4.69 | -0.26 | -1.32 | 0.88 | inconclusive | 

### Corpus control, provenance non-speech against main-bank non-speech

| corpus AUROC by duration (ms) | 20 | 40 | 60 | 80 | 100 | 200 | 500 | 1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unfiltered | 70.0 | 73.4 | 78.3 | 81.6 | 83.0 | 86.8 | 91.4 | 97.0 |
| filtered | 65.7 | 70.8 | 74.8 | 77.0 | 79.9 | 85.0 | 91.0 | 95.4 | 

