# Study design

The design below was fixed on 15 October 2026, before any score was extracted, and amended on 16 October, still before extraction, when the banks had to be regenerated (see [banks.md](banks.md)). The confirmatory run used it unchanged. Everything added later is exploratory and is recorded with its date in [exploratory_log.md](exploratory_log.md). The results of H1 to H4 are in [confirmatory_results.md](confirmatory_results.md).

The analysis library of this design is `src/erclm/lib_v2.py`. It was frozen with the design and has not changed since. Later analyses use `src/erclm/lib_v3.py`, which leaves the confirmatory readouts untouched.

## Model and stimuli

- Qwen2-Audio-7B-Instruct at revision `0a095220c30b7b31434169c3086508ef3ea5bf0a`, in NF4 with double quantisation and FP16 compute, batch size 1.
- Two backbones. The base model, and the base model with a task LoRA retrained with the recipe of the study under review (r = 64, α = 16, on the query, key, value and output projections) on the training bank. In the paper and in these documents the second one is the fine-tuned model.
- The forced-choice A/B prompt on which the LoRA was trained (`lib_v2.PROMPTS["ab"]`).
- A test bank (the main bank) of 970 base excerpts in 22 conditions, rebuilt from VoxConverse and ESC-50. Duration (8 levels) and SNR (6 levels) are the primary axes. Reverberation and low-pass filtering are secondary and lie outside the distribution the LoRA was trained on.
- SNR is measured on the excerpt, not on the container. Reverberation uses eight simulated rooms per level with measured T60 (0.3, 1.0 and 2.5 s).
- The training bank (64 excerpts by 48 duration-by-SNR cells), the validation bank (72 by 48) and the test bank share no recording.

## Readouts

Every readout starts from the score s = z_A − z_B, the difference between the logits of the two answer letters. The names below are the ones in the run records.

| Name | Readout | Role |
| --- | --- | --- |
| `L0` | The native decision, s > 0 | Reference |
| `L1_bc_cond` | Batch calibration within each condition: the threshold is the logit of the mean sigmoid of s over the unlabelled batch of that condition | Primary criterion readout |
| `L1_bc_global`, `L1_center_cond`, `L1_median_cond`, `L1_cf` | Batch calibration over all conditions, the per-condition mean of s, the per-condition median of s, and calibration on speech-free containers | Secondary variants |
| `oracle_ba` | The best threshold in each condition, chosen with the test labels | Upper bound only |
| `L2` | ℓ2-penalised logistic regression on the standardised frozen final state | Readout tested by H3 |
| `P_enc_global_mean`, `P_enc_region_mean` | Encoder probes on the global mean of the encoder frames, and on the regional mean of the frames that overlap the crop plus one frame of margin on each side | H2a |

- The confirmatory heads (`L2` and the encoder probes) are trained on the manifest the LoRA was trained on, the 3,072 variants of the training bank, with the inverse penalty C ∈ {10⁻⁴, 10⁻³, 10⁻², 10⁻¹} chosen by AUROC in four folds grouped by recording (`lib_v2.fit_head`).
- Heads cross-fitted inside the test bank are named `L2_cv` and `P_<feature>_cv`. They are exploratory and never enter H3 or H4 (`lib_v2.crossfit_scores`).
- The encoder probes also report d_a and the relative efficiency η of the native score with respect to the probe (`lib_v2.efficiency_table`).

## Hypotheses

- **H1.** Batch calibration (`L1_bc_cond`) raises the mean BA of the base model over the SNR axis.
- **H2a.** The AUROC of the regional encoder probe falls from 200 ms to 20 ms.
- **H2b.** The gain of L2 over L1 depends on duration. No sign was predicted.
- **H3.** The base model with L2 is not inferior to the fine-tuned model with L0. It is tested on each axis with the mean BA over the conditions of the axis. Non-inferiority requires the lower limit of the two-sided 95% interval to exceed −0.02 on the duration axis and on the SNR axis at the same time.
- **H4.** The gain of L2 over L1 is larger in the base model than in the fine-tuned model, as a difference of differences averaged over the conditions of each axis.

### Margin of H3

The margin of 2 BA points was fixed before extraction from two anchors in the study under review, neither of them a result of this study. Its LoRA configuration varied with a standard deviation of 2.4 BA points across the seeds of its prompt search, so a smaller difference is not practically distinguishable from that nuisance variation. Tripling its supervision, from 1,024 to 3,072 examples, changed BA by 0 to 1.8 points per condition, so a loss below 2 points is of the order of what three times more supervision buys. At constant specificity the margin equals 4 points of recall.

## Statistics

- Intervals come from a bootstrap over source recordings, stratified by class and paired across backbones, with 10,000 replicates in the confirmatory run (`lib_v2.cluster_bootstrap`).
- Label-free rules are re-estimated in each replicate from the resampled batch, with weights, so that the uncertainty of the threshold enters the intervals of H1 and H4. Readouts are carried as triples of score, threshold and rule for this purpose.
- The joint analysis aligns the same items of both backbones, and a single bootstrap resamples the same recordings for both (`lib_v2.combine_runs`). If the two backbones were identical, the difference of differences of H4 would be exactly zero in every replicate. A test checks this property.
- Each contrast reports its bootstrap standard error and `mde80` = 2.80 × SE, the smallest true effect detectable with 80% power at a two-sided 5% level. This is a post-extraction power analysis and does not modify any hypothesis.
- The prevalence check subsamples each condition to 20%, 50% and 80% speech and re-estimates the batch rules (`lib_v2.prevalence_check`). It is a secondary analysis of the balanced-batch assumption.
- Duration and SNR thresholds admit censoring and non-monotone curves (`lib_v2.threshold_table`).
- `top1_is_label` is a diagnostic, not a decision criterion.

## Scope

- The pilot of 50 excerpts per class only decides technical feasibility. Its contrasts are exploratory.
- The rise of false alarms at 1000 ms in Qwen2-Audio, absent in Qwen3-Omni, is an observation for the limitations.
- Training heads on the layers 8, 16 and 24 that are already extracted is allowed as an exploratory analysis.
- The closest prior work is Jo et al. (2026) and VocalGrad (Ryu and Yanaka, 2026). The contribution is to measure, condition by condition, how much of one detection decision a label-free recalibrated criterion, a linear readout and the LoRA recover.

## Exploratory extensions of 21 October

On 21 October, after the confirmatory results, two exploratory extensions were added to the closing plan. Neither changes H1 to H4, the confirmatory readouts, the metrics, the confirmatory prompt, NF4, batch size 1 or the seeds.

- **D1, prompt control.** Three prompt variants that cross the letter of the speech option with its position (`lib_v3.PROMPTS`), on 150 base excerpts per class of the duration and SNR axes, in both backbones.
- **D2, provenance bank v3** (`src/erclm/bankgen_v3.py`). Speech and non-speech windows from the same VoxConverse recordings, none of them used by the v2 banks, at the 8 durations and in both backbones.

## Timeline up to the confirmatory results

| Date (2026) | Step | Record |
| --- | --- | --- |
| 15 and 16 October | Design fixed. Smoke tests on Colab and Kaggle: NF4 takes 5.8 GB on a T4, the head reproduces the logits, 2 s of audio give 50 audio tokens and the output is deterministic | [`runs/smoke-v1`](../runs/smoke-v1) |
| 16 October | The banks and the adapter of the study under review turned out not to be accessible. The authors decided to regenerate the banks from VoxConverse and ESC-50 with corrections and to retrain the LoRA. Model revision pinned | [banks.md](banks.md) |
| 19 October | Banks v2 built: test bank with 21,340 variants (22 conditions, 970 base excerpts from 695 recordings, half speech), training bank with 3,072 variants of 64 excerpts and validation bank with 3,456 variants of 72, each in 48 cells | [`runs/banks-v2`](../runs/banks-v2) |
| 19 October | LoRA retrained in 576 steps. Validation loss 0.188 at step 200, 0.148 at 400 and 0.164 at 576, so checkpoint 400 was kept. 224 LoRA modules, 96 in the audio encoder and 128 in the language model, none of them zero | [`runs/retrain-v1`](../runs/retrain-v1) |
| 19 and 20 October | Pilot of both backbones on 100 base excerpts, 50 per class, feasibility only | [`runs/pilot-v1-base`](../runs/pilot-v1-base), [`runs/pilot-v1-lora`](../runs/pilot-v1-lora), [`runs/pilot-v1`](../runs/pilot-v1) |
| 20 October | Confirmatory run. Base model on a Colab T4 in 4 h 20 min, fine-tuned model on Kaggle, joint analysis on Kaggle CPU, each in a single attempt | [`runs/full-v1-base`](../runs/full-v1-base), [`runs/full-v1-lora`](../runs/full-v1-lora), [`runs/full-v1`](../runs/full-v1) |
| 21 October | Confirmatory results read. Exploratory closing plan starts | [confirmatory_results.md](confirmatory_results.md), [exploratory_log.md](exploratory_log.md) |
