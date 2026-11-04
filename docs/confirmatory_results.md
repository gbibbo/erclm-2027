# Confirmatory results

The confirmatory run `full-v1` ran on 20 October 2026 with the design of [study_design.md](study_design.md), unchanged. It scored 21,340 items from 695 source recordings of the main bank with both backbones. Intervals are 95% cluster-bootstrap intervals over source recordings with 10,000 replicates and seed 42, and the H3 margin is 2 BA points.

The paper reports H1, H2a and H3. H2b and H4, the secondary hypotheses, are reported here with the other three. Values are in points (BA or AUROC times 100). The raw values, as fractions, are in the files named in the last column.

| Hypothesis | Contrast | Estimate [95% CI] | Outcome | Record |
| --- | --- | --- | --- | --- |
| H1 | Base model, L1 minus L0, mean BA over SNR | +31.0 [29.3, 32.5] | Holds | `runs/full-v1-base/results/base/contrasts.csv`, `H1_snr_mean_BC_minus_L0_ba` |
| H2a | Regional encoder probe of the base model, AUROC at 200 ms minus 20 ms | +29.8 [26.2, 33.5] | Holds | same file, `H2a_S1_auroc_200_minus_20` |
| H2b | Base model, (L2 − L1) at 20 ms minus (L2 − L1) at 200 ms, BA | −4.0 [−8.3, 0.5] | No dependence on duration detected | same file, `H2b_(L2-BC)_20_minus_200` |
| H3, duration | Base L2 minus fine-tuned L0, mean BA | −3.6 [−4.9, −2.2] | Not shown, lower limit below −2 | `runs/full-v1/results/joint/joint_contrasts.csv`, `H3_duration_baseL2_minus_loraL0_ba` |
| H3, SNR | Base L2 minus fine-tuned L0, mean BA | −2.3 [−3.0, −1.7] | Not shown, lower limit below −2 | same file, `H3_snr_baseL2_minus_loraL0_ba` |
| H4, duration | (L2 − L1) in the base model minus (L2 − L1) in the fine-tuned model, mean BA | +1.2 [−0.3, 2.8] | Not shown | same file, `H4_duration_did_L2_minus_BC_ba` |
| H4, SNR | Same contrast over SNR | +14.6 [13.1, 16.3] | Holds | same file, `H4_snr_did_L2_minus_BC_ba` |

H3 required both lower limits above −2 at the same time, so it is not shown on either axis (`h3_noninferior_all_axes: false` in `runs/full-v1/results/joint/joint_summary.json`).

## Means per axis

Mean over the 8 durations (clean) and the 6 SNR levels (white noise) of the main bank, from the condition tables `runs/full-v1-base/results/base/condition_table.csv` and `runs/full-v1-lora/results/lora/condition_table.csv`. "Best" is the BA of the best threshold of each condition, chosen with the test labels (`oracle_ba`). These are the first five rows of Table 1 of the paper and the L2 row of the fine-tuned model, which the paper does not show.

| Backbone and readout | BA, duration | AUROC, duration | Best, duration | BA, SNR | AUROC, SNR | Best, SNR |
| --- | --- | --- | --- | --- | --- | --- |
| Base, L0 | 50.1 | 87.5 | 81.5 | 50.1 | 92.0 | 86.4 |
| Base, L1 | 79.1 | 87.5 | 81.5 | 81.1 | 92.0 | 86.4 |
| Base, L2 | 80.6 | 87.5 | 81.4 | 95.0 | 98.6 | 96.2 |
| Fine-tuned, L0 | 84.2 | 92.7 | 87.7 | 97.4 | 99.3 | 98.0 |
| Fine-tuned, L1 | 85.1 | 92.7 | 87.7 | 97.4 | 99.3 | 98.0 |
| Fine-tuned, L2 | 85.3 | 92.2 | 87.5 | 96.7 | 98.9 | 97.7 |

The probe rows of Table 1, marked as exploratory in the paper, come from heads cross-fitted inside the main bank. They are documented in [exploratory_log.md](exploratory_log.md), "Step 12. Accessibility per layer and heads trained inside the test bank" and "Numbers cited by the writing plan".

## Other pre-registered outputs

From the same two files.

| Contrast | Estimate [95% CI] |
| --- | --- |
| Base model, L1 minus L0, mean BA over duration | +29.0 [27.9, 30.2] |
| Fine-tuned model, L1 minus L0, mean BA over duration | +0.9 [0.6, 1.2] |
| Fine-tuned model, L1 minus L0, mean BA over SNR | +0.03 [−0.02, 0.10] |
| Base model, best threshold minus L0, mean BA over duration | +31.4 [30.5, 33.0] |
| Base model, L2 minus L1, mean BA over duration | +1.5 [−0.05, 3.0] |

The prevalence check (`runs/full-v1-base/results/base/prevalence.csv`) subsamples each condition to 20%, 50% and 80% speech. L1 assumes a balanced batch, and with 80% speech the base model's mean BA under L1 falls from 79.1 to 73.3 over duration and from 81.1 to 73.6 over SNR, while the best threshold stays at 81.9 and 86.9. The per-condition median (`L1_median_cond`) degrades less at 80% (76.9 and 77.4) and more at 20% (74.4 and 79.6).

## What the results say

- **H1.** The base model's score ranks speech above non-speech (AUROC 87.5 and 92.0), yet its native decision stays at 50.1 BA in every condition because it almost never answers "speech". Re-estimating the threshold per condition without labels recovers most of the distance to the best threshold. The failure is mostly one of criterion (evidence: AUROC against BA in the condition tables). Why the native threshold is so far off is examined in the exploratory log ("Prompt control", "Is the dominant failure the operating point?").
- **H2a.** Trained on the training bank, the encoder probe loses 29.8 AUROC points from 200 to 20 ms. Cross-fitted inside the main bank it loses far less, so the size of the fall depends on the probe's training data as well as on what the encoder holds (exploratory log, "Numbers cited by the writing plan").
- **H2b.** No dependence of the gain of L2 over L1 on duration was detected. The interval is wide (minimum detectable effect about 6 points, `mde80` in the contrasts file).
- **H3.** The linear readout trained on the fine-tuning data does not reach the fine-tuned model's native decision. On duration its failure is one of ranking, since its AUROC (87.5) is no higher than that of the native score. The exploratory log shows that the gap on duration depends on the penalty chosen by cross-validation on the noisy training bank and that no penalty would meet H3 under noise ("Sensitivity of the matched-supervision head to regularisation").
- **H4.** Under noise, L2 gains much more over L1 in the base model than in the fine-tuned model. The test SNR conditions coincide with cells of the training bank, which carries noise in every container, and there L2 ranks almost like the fine-tuned model (AUROC 98.6 against 99.3), while the fine-tuned model's L1 is already near its best threshold (inference from the table above). On the clean duration axis, outside the training distribution of L2, the difference is not shown.

## Recomputing these numbers

The per-item score tables of both backbones ship in the run records, so the joint analysis can be recomputed without a GPU:

```python
import pandas as pd
from erclm import analysis, export

tables = {b: export.load_scores(f"runs/full-v1-{b}/results/{b}/scores.csv.gz") for b in ("base", "lora")}
out = analysis.joint(tables, B=10000, seed=42, h3_margin=0.02)
print(out["tables"]["joint_contrasts"])
```

Run it from the repository root with `PYTHONPATH=src`. It reproduces `runs/full-v1/results/joint/joint_contrasts.csv` to within 10⁻¹⁶ and takes about 25 minutes on a laptop CPU with 10,000 replicates. The per-backbone files (condition tables, contrasts, prevalence) are written by the runner's analysis stage from the same score tables (`configs/full.yaml`, see [running.md](running.md)).
