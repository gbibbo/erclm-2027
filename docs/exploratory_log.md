# Exploratory log

This log records every exploratory analysis run after the confirmatory run of 20 October 2026, in the order it was done and with its date. None of them is a pre-registered test. When a section says that a rule was written before extraction, that rule was fixed before the data it reads were computed. The pre-registered design is in [study_design.md](study_design.md) and the results of H1 to H4 are in [confirmatory_results.md](confirmatory_results.md).

Intervals are 95% intervals from a cluster bootstrap over source recordings, stratified by class and paired across readouts and across backbones, with 2,000 replicates. Rules estimated from an unlabelled batch are re-estimated in each replicate. The two backbones are the base model and the fine-tuned model, which carries the LoRA. Readout names (L0, L1, L1*, L2 and the exploratory variants) follow the study design. BC is batch calibration, and BC per condition is the readout L1. Labels such as "Step 8" or "D1" refer to the closing plan summarised below. Drafts v3, v4 and v5 are earlier versions of the paper, and "the paper" inside a dated section means the draft of that date.

The tables of the first sections come from `tools/closing_analyses.py`, which reads the bank metadata, the VoxConverse RTTM files, the extracted features and the per-item scores of the run records in `runs/`.

```
PYTHONPATH=src python tools/closing_analyses.py v1
PYTHONPATH=src python tools/closing_analyses.py v2
PYTHONPATH=src python tools/closing_analyses.py tr --B 2000
PYTHONPATH=src python tools/closing_analyses.py anc --B 2000
PYTHONPATH=src python tools/closing_analyses.py cls --B 2000
```

## Closing plan of 21 October

The closing plan was written on 21 October 2026, after the review of draft v3 and an audit of that review. It followed the order the audit proposed, which corrects the claims first, then checks the validity of the stimuli and compares the readouts last. H1 to H4, the confirmatory readouts, the metrics, the prompt, NF4, batch size 1, the seeds and `src/erclm/lib_v2.py` stayed unchanged. Everything new was exploratory, had its own name and went into `src/erclm/lib_v3.py` with tests.

- **D1, prompt control.** Cross the letter that labels speech with the order of the options, on a subset and on both backbones. It changes the frozen prompt, so it enters only as exploratory. Adopted.
- **D2, optional GPU run.** Either a provenance control, with speech and non-speech from the same unused VoxConverse recordings, or E1 (an ablation over four states of the LoRA), or neither. The provenance control was adopted because it addresses the risk of a corpus shortcut. E1 was left for after submission.
- **D3 to D5.** Citation and submission logistics.
- **Steps 1 and 2.** Record the plan and take decisions D1 to D5.
- **Step 3.** Copy the metadata of the three banks, the VoxConverse RTTM files at commit `24bf60be297701cd7e4ef18550c6d390c1b87365` of `github.com/joonson/voxconverse` (the commit of the protocol) and the features of the full run, about 2.3 GB, and check the counts against the run records (21,340 test items and 3,072 training items per backbone).
- **Step 4 (V1).** For each of the 485 speech clips and each duration, measure what fraction of the central crop the RTTM covers, because the protocol requires 90% speech in the 1000 ms window and the centre can fall in a pause of up to 100 ms. Add a sensitivity analysis without the crops with less than 50% speech. If more than 5% of the 20 or 40 ms crops have no speech, report it as a limitation.
- **Step 5 (V2).** All speech comes from VoxConverse, all non-speech from ESC-50, and the noise of the SNR axis scales with the excerpt RMS. Fit a logistic regression on non-speech features (excerpt RMS, applied gain, noise standard deviation and peak from the metadata), cross-validated by recording on the training bank and evaluated per condition on the test bank like L2. Add crest factor and padding statistics from the audio only if needed. If AUROC exceeds 0.75 in any condition, the paper says that level cues are available, reads the probes with that caveat, and the provenance control becomes the priority run.
- **Step 6.** Code the GPU steps of D1 and D2. For D1, the four combinations of letter and order on 300 base clips over the 14 duration and SNR conditions, 4,200 items per variant. For D2, a provenance bank from the 170 unused VoxConverse recordings at the 8 durations, with the RMS criterion of ESC-50 to avoid silence, falling back to E1 if there were not enough non-speech windows with energy for about 150 clips.
- **Step 7.** Run the GPU jobs, estimated at about 2 T4 hours per backbone for D1 and less than 1 hour per backbone for the provenance bank. They run in the background of Steps 8 to 10.
- **Step 8 (L1_tr).** Fit one scalar threshold on s on the training bank, freeze it and apply it to the test bank. Compare it per condition with L0, global BC, BC per condition, centring, empty containers and L1*, with a bootstrap paired by recording, and report the means of s per class and condition. It replaces a constant-shift diagnosis.
- **Step 9 (L2_anc).** A head anchored to the native direction, s_anc = α s + δᵀh + b with α positive and an ℓ2 penalty on δ, fitted by grouped cross-validation on the training bank and compared per condition with L2 and with fine-tuned L0 through the H3 contrast, as exploratory.
- **Step 10.** False alarms per ESC-50 class, with non-verbal human vocal sounds apart from other human sounds and a bootstrap by recording, because the 22 variants of a clip are not independent. The full negative class stays in the primary analysis.
- **Step 11.** Analyse the GPU runs. For D1, whether the base model's bias changes with the letter or with the order, and whether the fine-tuned model transfers to the variants. For the provenance bank, how much each readout drops when speech and non-speech come from the same recordings. Runs not finished by 1 November stay out.
- **Step 12.** Margin. If a concrete question remains, such as whether the evidence is more accessible in an intermediate layer than in the final state, train probes on layers 8, 16 and 24, which were already extracted.
- **Step 13.** Freeze the analyses on 2 November. What is not finished stays out of the paper.
- **Steps 14 to 18.** Rewrite the paper and submit it.
- **Left for after submission.** Dense extraction per layer and per token (E2), token replacement (E3), a bank v3 with hypotheses H5 to H7 (E5), real noise and lower SNRs (E4) beyond the provenance control, E1 unless D2 chose it, an IAAN-style intervention, a second model, more LoRA seeds, and the additive decomposition proposed by the review of draft v3 (T2). The last one is replaced by three separate diagnostics, the threshold error, the ranking gain and the decodability of the probes.

## V1. Speech occupancy of the short crops

Done on 21 October (Step 4). According to the RTTM, the speech label of the short crops is correct. About 9% of the speech crops of 20 and 40 ms have very little energy, probably brief pauses that the RTTM marks as speech, and they are almost always missed. Removing them raises BA by at most 1.7 points per condition and changes no conclusion. It goes into the paper as one sentence in the limitations, with the sensitivity table in the appendix.

**RTTM coverage.** 99.8% of the speech crops from 20 to 100 ms fall entirely within annotated speech, and none has less than half covered. At 1000 ms the minimum coverage is 90%, as the protocol requires.

**Crop energy.** This is the RMS of the crop relative to that of the 1000 ms clip it comes from.

| Duration (ms) | Speech below −20 dB | Speech, 10th percentile (dB) | Non-speech below −20 dB | Non-speech, 10th percentile (dB) |
| --- | --- | --- | --- | --- |
| 20 | 9% | −18.9 | 15% | −26.7 |
| 40 | 8% | −18.5 | 14% | −25.5 |
| 60 | 7% | −17.5 | 13% | −24.4 |
| 80 | 5% | −15.7 | 12% | −23.1 |
| 100 | 4% | −14.4 | 12% | −22.6 |
| 200 | 2% | −8.2 | 9% | −18.1 |
| 500 | 0% | −2.6 | 4% | −9.2 |

Speech crops more than 20 dB below their clip, or with less than half covered by the RTTM, were flagged as doubtful. They are 172 of the 3,880 speech crops of the duration axis, 81 of them at 20 or 40 ms. At those durations the fine-tuned model with L0 gets 6% of the doubtful crops right against 39% of the rest, and the base model with L2 31% against 53%.

**Sensitivity.** BA in percent on the duration axis, with all items and without the doubtful speech crops. The thresholds are the stored ones.

| Condition | Base BC | Base BC without doubtful | Base L2 | Base L2 without doubtful | Fine-tuned L0 | Fine-tuned L0 without doubtful |
| --- | --- | --- | --- | --- | --- | --- |
| 20 ms | 63.5 | 63.1 | 64.3 | 65.1 | 61.9 | 63.0 |
| 40 ms | 75.9 | 77.2 | 71.2 | 72.2 | 72.7 | 74.2 |
| 60 ms | 80.1 | 81.4 | 76.0 | 76.9 | 81.5 | 83.2 |
| 80 ms | 81.1 | 82.5 | 76.9 | 77.6 | 82.5 | 83.9 |
| 100 ms | 82.2 | 83.0 | 80.8 | 80.9 | 85.8 | 86.9 |
| Axis mean | 79.1 | 79.7 | 80.6 | 81.1 | 84.2 | 85.1 |

The H3 gap on duration barely moves. With all items it is 80.6 against 84.2, and without the doubtful crops 81.1 against 85.1.

## V2. Level shortcut

Done on 21 October (Step 5). Level separates the classes little. The excerpt RMS alone gives an AUROC of 0.545 to 0.582 on the duration, SNR and reverberation axes, and 0.70 with the band-pass filter, always with speech louder. This stays below the plan's threshold of 0.75, so there is no need to download the audio to measure crest factor or padding statistics.

| Axis | AUROC of the excerpt RMS |
| --- | --- |
| Duration | 0.545 (20 ms) to 0.582 (500 ms) |
| SNR | 0.576 at every level |
| Reverberation | 0.576 at every level |
| Filtering | 0.576 unfiltered, 0.616 low-pass, 0.650 high-pass, 0.697 band-pass |

On SNR and reverberation the value does not change because the excerpt is the same 1000 ms clip. On the SNR axis the noise standard deviation is the excerpt RMS times a constant for each level, so the padding adds no level information within a condition.

V2 does not rule out channel or spectral cues specific to each corpus. VoxConverse comes from online videos and ESC-50 from freesound recordings. Only the provenance control of decision D2 settles that.

## Step 8. L1_tr, a threshold learned on the training bank

Done on 21 October. In the base model, a single threshold learned on the training bank and frozen matches or beats BC per condition on all four axes, and comes within 0.5 to 1.8 points of the best threshold of each condition. The base model's criterion problem is a misplaced threshold that transfers across conditions. Adjusting it condition by condition adds nothing beyond that.

The threshold is the one that maximises BA over the 3,072 training variants. In the base model it is −3.14, close to the −3.29 of the empty containers and far from the 0 of the native rule. In the fine-tuned model it is −0.84.

Mean BA of the base model per axis, in percent.

| Readout | What it needs | Duration | SNR | Reverberation | Filtering |
| --- | --- | --- | --- | --- | --- |
| L0 | Nothing | 50.1 | 50.1 | 50.2 | 50.3 |
| BC per condition (L1) | One unlabelled batch per condition | 79.1 | 81.1 | 85.2 | 82.3 |
| Global BC | One unlabelled batch | 74.8 | 80.8 | 87.3 | 84.3 |
| Centring per condition | One unlabelled batch per condition | 80.8 | 85.7 | 89.8 | 87.0 |
| Empty containers | 9 containers without content | 76.8 | 84.6 | 85.9 | 84.7 |
| L1_tr | Fixed threshold learned on the training bank | 79.7 | 85.9 | 88.4 | 86.4 |
| L1* | Test labels | 81.5 | 86.4 | 90.0 | 87.3 |

Paired differences in the base model, in BA points.

| Contrast | Duration | SNR | Reverberation | Filtering |
| --- | --- | --- | --- | --- |
| L1_tr − BC per condition | +0.6 [−0.3, 1.5] | +4.8 [3.8, 5.9] | +3.2 [1.9, 4.7] | +4.0 [2.5, 5.4] |
| L1_tr − L1* | −1.8 [−2.8, −1.5] | −0.5 [−1.2, −0.4] | −1.6 [−2.7, −1.1] | −0.9 [−2.0, −0.6] |
| BC per condition − global BC | +4.3 [3.3, 5.2] | +0.3 [−0.4, 0.7] | −2.1 [−3.0, −1.6] | −2.0 [−2.6, −1.3] |

In the fine-tuned model, L1_tr cannot be told apart from BC per condition on duration (−0.1 [−0.3, 0.1]) and falls 0.5 [0.2, 0.7] below it on SNR.

**Means of s per class.** In the base model the mean of non-speech stays between −3.39 and −3.88 in every duration and SNR condition. That of speech goes from −3.22 at 20 ms to −2.08 at 1000 ms, and on SNR stays between −2.28 and −2.37. The best threshold moves along the duration axis because speech moves, not non-speech.

**For the paper.** This replaces the sentence about a criterion that drifts with degradation. One possible wording is *A single threshold fitted on the training bank and frozen does as well as condition-wise batch calibration on every axis, and comes within 0.5 to 1.8 points of the best threshold per condition.* H1 stays as it is.

## Step 9. Heads anchored to the native direction

Done on 21 October. Anchoring the head to the native direction does not close the gap with the fine-tuned model. Two variants were tried.

- **L2_anc.** s_anc = α s + δᵀz + b, with α free and an ℓ2 penalty only on δ, as the audit proposed. In the base model the fit gives α = −0.14, so it practically discards the native direction.
- **L2_res.** s + δᵀz + b, with α fixed at 1. The penalty pulls towards the native direction and not towards zero. Grouped cross-validation chooses C = 0.01, the same as L2, and the correction stays large (‖δ‖ = 1.65).

In the fine-tuned model both choose C = 10⁻⁴ and stay almost on the native direction (‖δ‖ = 0.05).

Mean BA per axis, in percent.

| Readout | Duration | SNR | Reverberation | Filtering |
| --- | --- | --- | --- | --- |
| Base L2 | 80.6 | 95.0 | 94.4 | 95.2 |
| Base L2_anc | 80.6 | 95.0 | 94.3 | 95.2 |
| Base L2_res | 80.8 | 95.0 | 94.5 | 95.4 |
| Fine-tuned L0 | 84.2 | 97.4 | 96.5 | 97.4 |

Difference from fine-tuned L0, in the same format as H3 but exploratory.

| Base readout | Duration | SNR |
| --- | --- | --- |
| L2_anc − fine-tuned L0 | −3.6 [−4.9, −2.3] | −2.3 [−3.0, −1.7] |
| L2_res − fine-tuned L0 | −3.4 [−4.7, −2.1] | −2.4 [−3.0, −1.8] |

In AUROC the gap on duration is 5.0 points (87.7 against 92.7), so it is not a threshold problem.

**For the paper.** Neither the specified head nor one anchored to the native direction reaches the fine-tuned model. With the training data, which always carry noise, cross-validation prefers to move away from the native direction. That this preference comes from the distribution shift between noisy training and clean test remains a hypothesis.

## Step 10. False alarms per ESC-50 class

Done on 21 October. Vocal human sounds concentrate the false alarms. They are crying baby, sneezing, breathing, coughing, laughing and snoring, 62 of the 485 non-speech clips. The axes are duration and SNR, and the intervals come from resampling by recording.

| Readout | Vocal human | Non-vocal human | Rest |
| --- | --- | --- | --- |
| Base with BC per condition | 21.4% [15.9, 27.2] | 5.6% [3.0, 8.6] | 5.9% [4.9, 7.0] |
| Base with L2 | 19.2% [13.8, 25.2] | 7.0% [4.4, 10.0] | 9.5% [8.1, 11.1] |
| Fine-tuned with L0 | 10.4% [6.5, 14.9] | 1.0% [0.0, 2.9] | 1.5% [0.9, 2.1] |

In the fine-tuned model sneezing reaches 27% and coughing 11%. Crying baby, laughing, snoring and breathing stay between 0.6% and 6.5%.

**For the paper.** The full negative class stays in the primary analysis. A sentence in the limitations says that non-verbal vocalisations concentrate the false alarms, with the table in the appendix. It also notes that the prompt describes class B as "music/noise/silence/animals".

## Step 12. Accessibility per layer and heads trained inside the test bank

Done on 21 October with `tools/layer_accessibility.py`. For each backbone and each stored feature a linear head was fitted in two ways. The matched head is trained on the fine-tuned model's training bank, like L2. The cross-fitted head (cv) is trained inside the test bank with 5 folds grouped by recording, which on the final state is L2_cv.

Mean BA on the duration axis, in percent. The fine-tuned model with L0 gives 84.2.

| Base feature | Matched | Cross-fitted |
| --- | --- | --- |
| Encoder, global mean | 83.5 | 91.6 |
| Encoder, regional mean | 81.6 | 91.8 |
| Layer 8 | 70.1 | 91.4 |
| Layer 16 | 82.1 | 91.4 |
| Layer 24 | 81.2 | 90.8 |
| Final state | 80.6 | 90.7 |

On the main bank L2_cv beats the fine-tuned model, by +6.5 [5.5, 7.5] on duration and +1.1 [0.6, 1.8] on SNR. At 20 ms its AUROC is 0.89 against 0.80. With the cross-fitted head the regional encoder probe gives 0.91 at 20 ms, against 0.66 for the matched one.

**In-distribution curve.** The cross-fitted head was repeated with each training fold cut down to n base excerpts, half per class and one per recording, with C = 0.01 and three repetitions. With the base model's final state, 16 excerpts already give 85.0 BA on duration, 64 give 87.1 and the full set 90.7. The amount of supervision did not explain the difference from the matched heads.

**Reading.** "The provenance control of Step 11 shows that much of this advantage is a shortcut of the corpus pair." For that reason the cross-fitted heads are not used as evidence that the frozen state outperforms the fine-tuned model.

[REFUTED 2026-10-22 by the blind listening to the provenance negatives. With clean negatives and the same supervision, the in-bank advantage agrees between banks (+0.61 [−0.37, 1.74] against 0.98), and over all durations no difference remains to attribute to a corpus shortcut. The advantage comes from the amount of in-distribution supervision (section "Blind listening to the provenance negatives", joint reading). At 20 and 40 ms a residual of 2.43 [−0.22, 5.01] remains, with the mechanism open (N3, section "CPU analyses requested by the review of draft v5").]

Added on 22 October. With 182 excerpts per fold, as in the provenance bank, the AUROC advantage over fine-tuned L0 halves (section "Result of the matched-supervision reference").

## Step 11. Provenance control

Done on 21 October with `tools/closing_gpu.py prov`. The bank has 114 VoxConverse recordings not used in the v2 banks, one speech clip and one non-speech clip per recording, 228 clips and 1,824 variants over the 8 durations. Speech and non-speech share recording, channel and corpus. The matched heads and the L1_tr threshold are those of the main bank. The "testfit" heads are trained on the whole main bank (VoxConverse speech against ESC-50), without seeing the provenance bank.

**The native score and the matched heads do not depend on the corpus.** The mean AUROC over the 8 durations barely changes.

| Readout | Main bank | Provenance |
| --- | --- | --- |
| Base L0 | 87.5 | 88.7 |
| Base L2 matched | 87.5 | 87.5 |
| Fine-tuned L0 | 92.7 | 92.9 |
| Base L2 testfit | 95.8 | 91.4 |

**The heads trained on the corpus pair lose their advantage.** In mean BA they drop between 11 and 13 points (base L2 testfit from 90.7 to 77.9) and fall below the fine-tuned model. The drop is concentrated at the durations from 200 to 1000 ms.

Paired contrasts on the provenance bank, in points, with 2,000 replicates by recording.

| Contrast | AUROC 20 to 100 ms | AUROC 200 to 1000 ms | BA over all durations |
| --- | --- | --- | --- |
| Base L2 matched − fine-tuned L0 | −6.8 [−10.0, −3.7] | −3.2 [−5.3, −1.2] | −3.6 [−6.3, −0.8] |
| Base L2 testfit − fine-tuned L0 | −1.3 [−3.5, 0.9] | −1.9 [−3.5, −0.3] | −4.4 [−7.1, −1.7] |
| Base L2 testfit − base L2 matched | +5.5 [1.6, 9.3] | +1.3 [−0.7, 3.4] | −0.8 [−4.4, 2.6] |

At 20 ms the base model's regional encoder probe gives 65.2 AUROC if matched and 85.8 if trained on clean crops of the main bank. On the main bank it gave 66.3 and 90.8.

**For the paper.**
- The readout gap of H3 replicates with speech and non-speech from the same recordings. The matched head stays 3.6 BA points below the fine-tuned model, as on the main bank.
- Heads trained on the evaluation bank learn cues of the corpus pair. They beat the fine-tuned model inside that bank and fall below it on the provenance bank. This is a warning about probes trained on banks where each class comes from one corpus.
- At short durations, training the head on clean crops improves the ranking on the provenance bank too and brings it to the level of the fine-tuned model. The drop of H2a depends on how the probe is trained, not only on the representation.
- On the provenance bank the fine-tuned model's scores drop at long durations (AUROC of L0 at 1000 ms from 99.7 to 96.4). VoxConverse non-speech includes music and perhaps singing or unannotated background speech. This is a limitation of the control.

## Decision rule for the contrast inside the provenance bank

Written on 21 October before computing the contrast, following the audit of the exploratory findings. It decides whether the advantage of heads trained in distribution reflects speech evidence or cues of the corpus pair.

- **Primary estimand.** Difference in mean AUROC over the 8 durations between a head on the base model's frozen final state, cross-fitted inside the provenance bank (5 folds grouped by recording, C chosen by inner cross-validation), and the fine-tuned model with L0. Bootstrap paired by recording with 2,000 replicates.
- **Reference.** The same difference on the main bank without the crops below −30 dBFS, which is 2.36 points.
- **Speech evidence.** The lower limit is above 0 and the estimate is at least 1.18 (half the reference).
- **Shortcut.** The estimate is below 0.79 (a third of the reference) and the upper limit is below 2.36.
- **Inconclusive.** Any other result.
- **Secondary contrast.** The same rule on the mean of 20 and 40 ms, with a reference of 7.65 points (thresholds 3.83 and 2.55). It has more power.

## Contrast inside the provenance bank

Done on 21 October with `tools/closing_gpu.py provcv`, after the rule was recorded. The heads are cross-fitted inside the provenance bank, with 5 folds grouped by recording and C chosen by inner cross-validation in each fold.

**Decision under the rule. Shortcut in all four contrasts.** Inside the provenance bank the advantage of the in-distribution heads over the fine-tuned model disappears, and the upper limits exclude the effect sizes of the main bank.

| AUROC contrast | Estimate | 95% CI | Reference | Decision |
| --- | --- | --- | --- | --- |
| All durations, base cv − fine-tuned L0 | +0.14 | [−1.11, 1.47] | 2.36 | Shortcut |
| All durations, fine-tuned cv − fine-tuned L0 | +0.37 | [−0.75, 1.58] | 2.36 | Shortcut |
| 20 and 40 ms, base cv − fine-tuned L0 | −0.38 | [−2.34, 1.76] | 7.65 | Shortcut |
| 20 and 40 ms, fine-tuned cv − fine-tuned L0 | −0.13 | [−1.97, 1.98] | 7.65 | Shortcut |

Added on 22 October. The references in this table come from heads with full supervision. With the matched-supervision reference, the two contrasts over all durations become inconclusive and the two at 20 and 40 ms stay at shortcut (section "Result of the matched-supervision reference").

Other results inside the provenance bank.
- The final states of the base and fine-tuned models are equally decodable with in-distribution heads, −0.23 [−0.90, 0.36] AUROC.
- The matched head stays below the fine-tuned model, −5.43 [−7.79, −3.10] AUROC, and −7.74 [−11.77, −3.84] at 20 and 40 ms.
- The layer profile is flat. The mean AUROC goes from 93.0 to 93.8 across the encoder, layers 8, 16 and 24 and the final state.
- The drop of the regional encoder probe from 200 to 20 ms is 8.0 [4.5, 11.6] AUROC points, against 29.8 for the matched probe on the main bank.

**Corpus control.** A head separates VoxConverse non-speech (provenance) from ESC-50 non-speech (main bank) with AUROC of 70 to 77 at 20 ms and of 96 to 97 at 1000 ms. The corpus is linearly decodable at every depth, so it was available as a shortcut.

## Sensitivity of the matched-supervision head to regularisation

Done on 21 October with `tools/closing_gpu.py csweep`. The base model's L2 head was refitted on the training bank with each C of the pre-registered grid. Grouped cross-validation on that bank chooses C = 0.01, with AUROC 0.891 against 0.884 for C = 0.001 and 0.858 for C = 0.0001.

Difference from fine-tuned L0, in points.

| C | BA duration | BA SNR | AUROC duration | AUROC SNR | AUROC provenance |
| --- | --- | --- | --- | --- | --- |
| 0.0001 | +1.06 [0.13, 1.95] | −1.86 [−2.40, −1.33] | −0.84 [−1.20, −0.50] | −0.17 [−0.50, 0.15] | −0.61 [−1.38, 0.10] |
| 0.001 | +0.23 [−0.64, 1.07] | −1.65 [−2.18, −1.15] | −1.58 [−2.15, −1.03] | −0.26 [−0.57, 0.05] | −1.67 [−3.15, −0.31] |
| 0.01 (chosen) | −3.58 [−4.90, −2.32] | −2.34 [−3.00, −1.68] | −5.14 [−6.24, −4.10] | −0.68 [−1.05, −0.35] | −5.43 [−7.79, −3.10] |
| 0.1 | −9.37 [−10.95, −7.80] | −4.42 [−5.28, −3.56] | −11.31 [−12.95, −9.74] | −1.64 [−2.16, −1.20] | −13.60 [−17.19, −10.01] |

**Reading.** In scikit-learn C is the inverse of the regularisation strength, so C = 0.0001 is the strongest penalty of the grid and C = 0.1 the weakest. H3 still fails as pre-registered. With the strongest penalty of the grid it would fail only on SNR, and narrowly. Cross-validation on the training bank, which always carries noise, prefers a weakly regularised head that transfers poorly to the clean test. With C = 0.0001 the linear readout of the frozen final state comes within one AUROC point of the fine-tuned model on the main bank and on the provenance bank.

With C chosen by inner cross-validation in all cross-fitted heads (refit of 21 October), the fine-tuned model's layer profile on the main bank is also flat. The mean BA on duration goes from 90.7 to 92.0 across the encoder, layers 8, 16 and 24 and the final state. The earlier dip at layers 16 and 24 was an artefact of C.

## Audit of the exploratory findings

Done on 21 October. The full audit reports are not part of this record.
- **Leakage.** There is none. Folds never split a recording, a base clip or the three neutral versions, and every number reproduces with independent code.
- **The advantage of L2_cv on the main bank.** It comes out the same on the fine-tuned model's state (90.66). Half of it is threshold placement (AUROC +3.09, best-threshold BA +3.39). With 64 in-distribution excerpts the BA advantage drops to +2.4 to +2.9.
- **Channel cues on the main bank.**
  - VoxConverse has a band cut from resampling above 7.4 kHz that separates the corpora with AUROC of 0.66 to 0.76, even against the vocal sounds of ESC-50, and that the training noise erases.
  - The content below 40 Hz and the level of the pauses also differ.
  - A logistic regression on 64 log-mel bands of the crop gives 0.833 AUROC at 20 ms, and a gradient boosting classifier reaches 0.877.
- **Encoder preprocessing.** The Whisper front end clips the log-mel at the container maximum minus 8, so the padding frames carry the level and the spectral peak of the crop. The encoder's global mean is not a neutral average of the context, and the difference between the global and regional probes is not due only to how the tokens are pooled.
- **Tool errors.** The cross-fitted heads on features other than the final state fixed C at the matched value. For the fine-tuned model this leaves its layer profile on the main bank without interpretation. The profile inside the provenance bank already chooses C by inner cross-validation.
- **Analysis history.** The cross-fitted heads were promoted after H3 failed, and the pilot had already shown them within the margin. They are exploratory and must be presented as such.

## Reading rule for the prompt control

Written on 21 October, before opening the two variants that change the letter of the speech option (`ba_speech_first` and `ba_speech_second`). The variant that changes only the position (`ab_speech_second`) had already been looked at. In it the base model still answers speech for 1% of the speech items, and the fine-tuned model keeps its AUROC (91.3 against 91.9 on duration) and loses BA with its native rule (78.3 against 83.7).

- **Measures per variant and backbone, paired by item with the original prompt on the same 4,200 items.** Share of speech answers with L0 per class, mean AUROC of s over the duration conditions and over the SNR conditions, and BA with L0 and with L1.
- **Letter bias.** If, in the two variants with B as speech, the base model answers speech for at least 50% of the speech items, the base model's criterion failure is largely a preference for the letter B.
- **Label bias.** If in all four variants the base model answers speech for less than 5% of the speech items, the failure follows the label of the option and not its letter or its position.
- **Mixed.** Any other pattern is reported as mixed, with the shares.
- **Fine-tuned model.** If its mean AUROC on duration drops by more than 2 points with B as speech relative to the original prompt, its gain depends on the letter it was trained with. Otherwise its ranking is robust to the change of letter.

## Prompt control

Done on 21 October with `tools/closing_gpu.py prompt` on 300 base clips, 150 per class, of the duration and SNR axes. That is 4,200 items per variant and backbone, paired with the original prompt. The four variants are `ab` (the original, A is speech and comes first), `ab_speech_second` (A is speech and comes second), `ba_speech_first` (B is speech and comes first) and `ba_speech_second` (B is speech and comes second). In all of them the first token of the answer is A or B in 100% of cases.

Share of speech answers with L0, and mean AUROC of s on duration and on SNR.

| Variant | Base, speech answered on speech items | Base, speech answered on non-speech items | Base AUROC duration / SNR | Fine-tuned, speech on speech items | Fine-tuned, speech on non-speech items | Fine-tuned AUROC duration / SNR |
| --- | --- | --- | --- | --- | --- | --- |
| ab | 0.1% | 0.0% | 86.6 / 92.0 | 82.2% | 3.6% | 91.9 / 98.8 |
| ab_speech_second | 1.0% | 0.0% | 82.8 / 96.4 | 74.2% | 1.6% | 91.3 / 98.7 |
| ba_speech_first | 99.9% | 99.8% | 39.7 / 29.7 | 98.7% | 98.8% | 23.3 / 33.1 |
| ba_speech_second | 96.0% | 49.3% | 86.7 / 96.1 | 88.5% | 17.7% | 88.2 / 98.0 |

**Reading under the rule written before opening the variants with B as speech.**
- **Base model, letter bias.** With B as speech it answers speech for 99.9% and 96.0% of the speech items, more than the 50% the rule sets. The base model's criterion failure is largely a preference for the letter B. With `ba_speech_second` its BA with the native rule rises from 50.1 to 68.2 on duration and from 50.1 to 80.2 on SNR, without touching the threshold.
- **Fine-tuned model, dependent on the letter it was trained with.** With B as speech its AUROC on duration drops by 3.7 [2.1, 5.4] points in `ba_speech_second` and by 68.6 in `ba_speech_first`. Both drops exceed the 2 points of the rule.

**Description, outside the rule.** With the options in the order A, B, the ranking of both backbones follows the speech label, whatever its letter. With the order B, A, it follows the letter A, whatever its label, and AUROC falls below 50 when A names non-speech (base 39.7 and 29.7, fine-tuned 23.3 and 33.1). The original prompt hides both dependencies.

## Numbers cited by the writing plan

Computed on 21 October with `tools/closing_extra.py` and with the extended version of `tools/closing_gpu.py provcv`, with 2,000 replicates.

**Main bank, duration axis, base L2_cv against other readouts.**
- Against fine-tuned L0 it is +6.55 [5.52, 7.52] BA, +3.09 [2.28, 3.91] AUROC and +3.39 [2.44, 4.23] best-threshold BA.
- Against fine-tuned L2_cv it is +0.05 [−0.36, 0.45] BA and −0.06 [−0.25, 0.11] AUROC.
- On SNR, against fine-tuned L0, it is +1.12 [0.56, 1.75] BA and +0.53 [0.21, 0.89] AUROC.

**H4 in AUROC.** The difference in differences of L2 over L0 between backbones is 6.98 [5.72, 8.29] points on SNR and 0.52 [−0.90, 1.89] on duration.

**Drop of the regional encoder probe between 200 and 20 ms.**

| Probe | Main bank | Provenance bank |
| --- | --- | --- |
| Matched | 29.83 [26.14, 33.48] | 25.95 [18.51, 33.70] |
| Cross-fitted | 8.42 [6.74, 10.12] | 7.96 [4.46, 11.62] |

**Advantage of L2_cv over fine-tuned L0 per duration on the main bank, in AUROC points.** Without the crops below −30 dBFS it is 4.12 at 20 ms, 3.07 at 40 ms and 0.33 at 1000 ms. With all crops it is 9.66, 5.60 and 0.33.
- The secondary reference of the rule, 7.65, used all crops, while the primary one excluded the silent crops.
- Without the silent crops, the secondary reference is 3.60, with thresholds of 1.80 and 1.20.
- The observed secondary contrast, −0.38 [−2.34, 1.76], still classifies as shortcut.

**Extremes of the criterion on the main bank.**
- The largest share of speech answers of the base model with L0 in any condition is 0.82% (low-pass).
- The worst difference per condition between L1_tr and L1 is −1.96 points, at 100 ms. The best is +6.6.

**Level used in both directions.** A cross-validated logistic regression on the log RMS and its square gives an AUROC of 0.58 to 0.59 between 20 and 100 ms, 0.61 at 200 ms, 0.66 at 500 ms and 0.72 at 1000 ms.

**Provenance bank.**
- The primary contrast of the rule has a standard error of 0.66 and a minimum detectable effect of 1.86 points. The secondary one has 1.04 and 2.90.
- Base L2_cv against fine-tuned L0 gives +4.99 [3.18, 6.80] BA and +0.44 [−1.37, 2.19] best-threshold BA.
- L1* is 83.44 for the base model and 87.83 for the fine-tuned model.

AUROC difference of each depth of the base model (cv) against the final state and against fine-tuned L0, inside the provenance bank.

| Feature | Minus the final state | Minus fine-tuned L0 |
| --- | --- | --- |
| Encoder, regional mean | +0.72 [−0.27, 1.67] | +0.86 [−0.59, 2.42] |
| Encoder, global mean | +0.22 [−0.74, 1.07] | +0.36 [−1.10, 1.82] |
| Layer 8 | +0.01 [−1.03, 1.00] | +0.14 [−1.29, 1.53] |
| Layer 16 | −0.09 [−1.03, 0.77] | +0.05 [−1.05, 1.27] |
| Layer 24 | +0.09 [−0.71, 0.89] | +0.23 [−0.88, 1.34] |
| Final state | | +0.14 [−1.11, 1.47] |

**Corpus control at every depth read.** The cross-fitted AUROC between VoxConverse non-speech and ESC-50 non-speech goes from 70.0 to 77.3 at 20 ms and from 95.2 to 97.0 at 1000 ms. It was measured on the encoder (global and regional mean), on layers 8, 16 and 24 and on the final state.

**Figure.** `tools/make_figures.py --v4 abc` draws three exploratory panels, (a) the criterion, (b) the regularisation and (c) the provenance. The data come from `results/closing/csweep_contrasts.csv` and `results/closing/figure_auroc_diff.csv`.

## Erratum to the provenance rule

Found on 21 October by the review of draft v4. The recorded text of the rule describes the primary reference as the advantage of L2_cv without the crops below −30 dBFS, and gives it the value 2.36. That value comes from another exclusion, that of the crops more than 20 dB below their 1000 ms clip (550 crops out and 7,210 in). The description and the number do not match. The recorded text is not modified.

| Exclusion of silent crops | Primary reference | 20 and 40 ms reference |
| --- | --- | --- |
| None, all crops | 3.09 | 7.63 |
| More than 20 dB below their clip (the recorded number) | 2.36 | 5.80 |
| Below −30 dBFS (the recorded description) | 1.54 | 3.60 |

The recorded secondary reference, 7.65, is the mean of the advantages at 20 and 40 ms after rounding (9.7 and 5.6). Without rounding it is 7.63, with thresholds of 3.82 and 2.54. No verdict changes.

With any of the three references the verdict is the same. The primary contrast, +0.14 [−1.11, 1.47], has an estimate below a third of the reference and an upper limit below the reference. With 1.54 the thresholds are 0.51 and 1.54, and they are met narrowly. The secondary contrast, −0.38 [−2.34, 1.76], also gives shortcut with all three.

Added on 21 October. On the provenance bank, the fine-tuned model's contrast (+0.37 [−0.75, 1.58]) is inconclusive with the reference of 1.54, because its upper limit exceeds it. The two base contrasts give shortcut with all three definitions.

Added on 22 October. All three definitions use heads with full supervision. With matched supervision the base contrast over all durations is inconclusive with the 20 dB and the −30 dBFS exclusions and gives shortcut without exclusion (section "Result of the matched-supervision reference").

## Cross-validation of the matched-supervision head over C

Computed on 21 October on the 3,072 training variants of the base model, with 4 folds grouped by recording, as in `lib_v2.fit_head`.

| Fold seed | C = 0.0001 | C = 0.001 | C = 0.01 | C = 0.1 |
| --- | --- | --- | --- | --- |
| 0 (the one of the confirmatory fit) | 86.9 | 89.4 | 89.5 | 87.9 |
| 42 (the one of the C sweep) | 85.8 | 88.4 | 89.1 | 88.5 |

Both seeds choose C = 0.01. With seed 0 the preference over C = 0.001 is 0.1 point. The record of the confirmatory run (`runs/full-v1-base/results/base/fit_info.csv`) gives 91.0 at C = 0.01, a value that does not reproduce with either seed, and the cause has not been traced. The refitted head reproduces the confirmatory scores with correlation 0.999995.

## Pre-registered outputs the draft did not show

Read on 21 October from the run records `runs/full-v1/results/joint/joint_contrasts.csv` and `runs/full-v1-base/results/base/` (`contrasts.csv`, `prevalence.csv` and `thresholds.csv`).

**Standard error and minimum detectable effect (2.8 standard errors), in points.**

| Hypothesis | Standard error | Minimum detectable effect |
| --- | --- | --- |
| H1 | 0.80 | 2.25 |
| H2a | 1.85 | 5.17 |
| H2b | 2.23 | 6.24 |
| H3 on duration | 0.67 | 1.89 |
| H3 on SNR | 0.33 | 0.93 |
| H4 on duration | 0.80 | 2.25 |
| H4 on SNR | 0.81 | 2.27 |

**Prevalence.** With 20%, 50% and 80% speech in the batch, BC per condition gives 85.9, 81.1 and 73.6 on SNR, against 86.8, 86.4 and 86.9 for L1*.

**Thresholds.** BA reaches 90% on duration at 148 ms with fine-tuned L0 and at 296 ms with base L2. The threshold readouts of the base model do not reach it. On SNR almost all readouts exceed 75% even at −10 dB, so their threshold is censored from below.

## Source of the numbers in Table 1 of draft v4

All of them come from files in `runs/` or from the closing analyses, and match what the draft gives.
- **Main bank.** The BA and AUROC per axis come from `condition_table.csv` of each backbone, from `results/closing/csweep_contrasts.csv` (C = 0.0001) and from the accessibility report (L2_cv).
- **Mean AUROC on SNR of the base model with L0.** It is 92.0.
- **Provenance column.** It comes from `provcv_ci.csv` and `prov_ci.csv`, which `tools/closing_gpu.py provcv` and `prov` write for the run `prov-v3`. The values are 50.0, 78.2, 83.4, 78.7, 82.3, 83.0, 80.2, 83.8, 87.3, 77.9 and 87.2 in BA, and 88.7, 87.5, 92.9, 92.1, 92.3, 93.1, 91.4 and 93.3 in AUROC.
- **Fine-tuned L2 row on the main bank.** It is 85.3, 96.7, 92.2 and 98.9.
- **Range at 20 ms on provenance.** It goes from 85.9 to 87.4 across the L2_cv heads and fine-tuned L0.

## Reference with matched supervision

**Reading rule, written on 21 October before computing.** The review of draft v4 noted a difference in supervision. The L2_cv heads of the provenance bank train on about 182 clips per fold, from 91 recordings with one speech clip and one non-speech clip each, and only with the 8 durations. The reference of the main bank comes from heads trained on about 776 excerpts and all axes. This check recomputes the reference with the supervision of the provenance bank. It is a sensitivity analysis and does not change the recorded verdict.

- **Head.** Base L2_cv on the main bank, with the same outer folds (seed 42) and the same inner choice of C as `lib_v2.crossfit_scores`. Each training fold is cut to 182 excerpts, 91 per class and one per recording. The primary variant, "clips and durations", uses only their 8 duration variants, as in the provenance bank. The secondary variant, "clips", uses all their conditions.
- **Matched reference.** The mean over 10 draws of the mean AUROC advantage over fine-tuned L0 over the 8 durations, without the crops more than 20 dB below their clip, which is the exclusion that gave 2.36. The other two exclusions and the mean of 20 and 40 ms are also reported.
- **Reading.** The recorded rule is applied to the four contrasts with the matched reference. If the base contrast over all durations stays at shortcut, the paper says that the verdict withstands matched supervision. If it becomes inconclusive, the paper says that with matched supervision the tie inside the provenance bank is not enough to show the shortcut, and that the shortcut then rests on the drop of the transferred head and on the decodability of the corpus. In that case the abstract is also adjusted.
- **Code.** `python tools/closing_extra.py matched --draws 10`, step I.

## Result of the matched-supervision reference

Done on 21 October with `tools/closing_extra.py matched --draws 10`, after the rule was recorded, except for draw 0. That draw came from a single-draw test that finished one minute before the rule was recorded. The test filtered the table rows in the console, so it showed no reference, and its `references.csv` and `decisions.csv` were overwritten. The 10-draw run reused draw 0 and computed the other 9 after the rule was published. The recorded code reproduces draw 0 without differences.

Reference in AUROC points, mean of the 10 draws with the minimum and maximum in parentheses. Full supervision is that of the L2_cv heads that gave the recorded reference, with 776 excerpts and the 22 conditions in each training fold.

| Crop exclusion | Durations | Clips and durations (primary) | Clips | Full supervision |
| --- | --- | --- | --- | --- |
| None | All | 1.53 (1.30 to 1.87) | 1.51 (1.25 to 1.75) | 3.09 |
| None | 20 and 40 ms | 4.50 (3.64 to 5.56) | 4.14 (3.13 to 5.24) | 7.63 |
| More than 20 dB below their clip | All | 1.01 (0.75 to 1.38) | 0.99 (0.72 to 1.18) | 2.36 |
| More than 20 dB below their clip | 20 and 40 ms | 3.00 (2.38 to 4.06) | 2.65 (1.62 to 3.71) | 5.80 |
| Below −30 dBFS | All | 0.40 (0.11 to 0.71) | 0.44 (0.17 to 0.64) | 1.54 |
| Below −30 dBFS | 20 and 40 ms | 1.21 (0.50 to 2.03) | 1.06 (0.11 to 1.99) | 3.60 |

- **Decision with the primary matched reference.** It is 1.01, with thresholds of 0.50 and 0.34. The two contrasts over all durations are inconclusive, because their upper limits, 1.47 for the base model and 1.58 for the fine-tuned model, exceed it. The fine-tuned estimate, 0.37, also exceeds a third of the reference. The two contrasts at 20 and 40 ms stay at shortcut against 4.50, which uses all crops like the recorded secondary reference, and also against 3.00, with the 20 dB exclusion.
- **Supervision.** With 182 excerpts per fold the advantage of L2_cv over fine-tuned L0 on the main bank halves, from 3.09 to 1.53 without exclusion, and the reference goes from 2.36 to 1.01. The advantage stays concentrated at short durations, with 6.18 at 20 ms and 0.16 at 1000 ms without exclusion. The "clips" variant, with the 22 conditions of the same excerpts, gives almost the same, so what matters is the number of excerpts and not the number of conditions.
- **Chosen C.** The matched-supervision heads choose C = 0.001 in 36 of the 50 folds and 0.01 in 14, and the "clips" variant in 30 and 20. The provenance heads choose 0.001 in 3 of 5 and the full-supervision heads 0.01 in all 5. In the check, with C fixed at 0.01 the matched reference is 1.01, and with C fixed at 0.001 it is 1.14, so the drop comes from the number of excerpts and not from the choice of C.
- **Other exclusions.** The base contrast over all durations gives shortcut without exclusion, because its upper limit of 1.47 falls just below 1.53, and inconclusive with the other two exclusions. Without exclusion, the single-draw reference exceeds 1.47 in 5 of the 10 draws. The fine-tuned contrast is inconclusive with all three. The contrasts at 20 and 40 ms are inconclusive with the −30 dBFS exclusion (1.21). The "clips" variant gives the same decisions.
- **Reading under the rule.** The base contrast over all durations is inconclusive. With matched supervision the tie inside the provenance bank is not enough to show the shortcut, "which then rests on the drop of the transferred head and on the decodability of the corpus". At 20 and 40 ms, where the advantage was concentrated, both contrasts stay at shortcut against 4.50 and against 3.00. The verdict of the rule recorded on 21 October, with its own references, does not change. The abstract, sections 4.3 and 5 and the appendix of the draft are adjusted.
  [REFUTED 2026-10-22 by the blind listening to the provenance negatives. Without the 38 negatives with speech, the transfer drop of the same head is no longer detectable, and that the corpus is decodable does not show that the speech heads use it (section "Blind listening to the provenance negatives").]

**Verification.** Done on 21 and 22 October. The full reports are not part of this record.
- **Reproduction.** Refits of the 5 outer folds and of the 10 draws reproduce the stored scores with a maximum difference of 0 and the same C. An independent implementation of the references agrees up to machine rounding and gives the same 36 decisions.
- **Uncertainty of the reference.** With the heads fixed, the bootstrap by recording gives 1.01 [0.37, 1.65] for the primary reference. Only 6.5% of the replicates exceed 1.47 and would give shortcut for the base model. The difference between full and matched supervision is 1.35 [1.06, 1.65], outside the evaluation noise. Across draws the standard deviation is 0.17, and the mean of 30 draws is 1.05.
- **Sampling.** The draw shuffles the excerpts of each class, keeps the first of each recording and takes the first 91. So a speech recording with 3 excerpts enters with probability 0.64 and one with 2 excerpts with probability 0.50. Correcting this moves the primary reference by +0.02 ± 0.02, and a uniform draw by recording gives 1.04. No decision changes.
- **Asymmetries unrelated to supervision.** The folds of the provenance bank chose different values of C, and pooling their scores lowers the estimate, which favours the shortcut. With the same C fixed on both sides no base decision changes. Over all durations the base contrast is +0.42 [−0.62, 1.50] against a reference of 1.14 with C = 0.001, and +0.39 [−1.04, 1.86] against 1.01 with C = 0.01. In the opposite direction, and with all crops, fine-tuned L0 is further from the ceiling on the provenance bank, with AUROC of 93.9 to 96.4 between 200 and 1000 ms against 98.1 to 99.7 on the main bank. There the in-bank head sits between −0.38 and +1.22 above the fine-tuned model, while the advantage with matched supervision goes from −0.12 to +0.27, and that raises the contrast over all durations. Between 20 and 100 ms the provenance contrast goes from −0.76 to +0.38 and the advantage with matched supervision from 0.48 to 6.18.
- **Exclusion.** As in the recorded rule, the exclusion of silent crops applies only to the reference, because the files used for this analysis had no relative level for the provenance crops. The base verdict over all durations depends on that exclusion.
- **Tool.** The verification found two defects in `tools/closing_extra.py`, fixed on 22 October. The table of means and ranges in `report.md` shifted its columns when rendered, because its two-level index had no names. The cache of draws was keyed only by draw, n and seed, and now each draw stores its parameters and is recomputed if they change. The 10 draws were recomputed with the fixed code.

## GPU plan of 22 October

**Reading rules, written on 22 October before extraction.** The review of draft v4 objects that the paper does not tell a corpus shortcut apart from other distribution shifts, and that the drop of the transferred head compares two different heads. The authors chose to answer with GPU experiments, run on Colab. Both are exploratory and touch nothing frozen.

### I1. Intervention on channel cues

**What changes.** Every container the model hears first passes through a band-pass filter from 60 to 6000 Hz, Butterworth of order 8 with zero phase (`lib_v3.bandpass`). It removes the VoxConverse band cut above 7.4 kHz and the differences below 40 Hz that the audit found. It does not touch the level of the pauses or the content of the sounds. The configurations are `cue_main` (duration axis of the main bank, 7,760 variants) and `cue_prov` (provenance bank), with both backbones.

**Common procedure.** Variants of the duration axis ordered by `variant_id`, all crops, and heads on the base model's final state with `lib_v2.crossfit_scores` (5 folds by recording, seed 42, C by inner cross-validation), trained only on duration variants. The original version is recomputed with the same code on the features of `full-v1`, so the two versions share items, folds and procedure. Bootstrap paired by recording with 2,000 replicates. Numbers are given in points of mean AUROC over the 8 durations.

- **Estimand A, the in-bank advantage.** ΔA is the advantage of L2_cv over fine-tuned L0 with the filter minus the same advantage without the filter.
- **Estimand B, the transfer gap of the same head.** Each fold head is applied unchanged to its held-out fold of the main bank and to the whole provenance bank. The gap is the AUROC on the held-out fold minus the AUROC on provenance, averaged over the 5 heads. ΔB is the gap with the filter on both banks minus the gap without the filter. For its interval the recordings of each bank are resampled separately, with the heads fixed.
- **Reading of each estimand.** With the original advantage or gap O, the filter "explains it" if the upper limit of Δ is below 0 and the estimate is at most −O/2. It "does not explain it" if the lower limit of Δ is above −O/4. Any other result is inconclusive.
- **What the paper says.** If the filter explains A and B, the corpus shortcut is backed by an intervention and the cue is named. If it explains neither, the advantage and the drop do not come from those channel cues, the mechanism stays open and the paper adopts the reframing the review asks for. The reading is asymmetric. That the filter does not explain the advantage does not rule out other corpus cues, such as the level of the pauses or the content.
- **Control.** The change in AUROC of fine-tuned L0 and of the base model's native score with the filter is reported. If fine-tuned L0 moves by more than 2 points, the paper says that the filter also changed the difficulty of the task.
- **Descriptive.** The decodability of the corpus with the filter, with provenance non-speech against ESC-50 non-speech per duration, as in the corpus control of Step 11.

### I2. LoRA restricted to part of the model

**What changes.** The retrained LoRA stays active in only one part of the model and the rest of its modules are switched off (`lib_v3.restrict_lora`). The variants are the audio encoder (`split_audio`, 96 modules), the language model (`split_llm`, 128) and its layers 0 to 15 and 16 to 31 (`split_llm_lower` and `split_llm_upper`). They use the same 150 base clips per class of the prompt control, on the duration and SNR axes. The base model and the full LoRA come from `full-v1` for the same items.

- **Estimands.** On each axis, the fraction of the LoRA's gain that a variant v recovers. For ranking it is R(v) = [AUROC(v) − AUROC(base)] / [AUROC(LoRA) − AUROC(base)], with the native score and the mean over the conditions of the axis. For the criterion it is C(v), the same fraction with the BA of L0. Intervals come from a bootstrap paired by recording with 2,000 replicates, with the fraction computed in each replicate.
- **Reading, on the duration axis and for the pair of encoder and language model.** A part "carries" the gain if the lower limit of its fraction is above 0.5 and the upper limit of the other part is below 0.5. If both upper limits are below 0.5, both parts are needed. If both lower limits are above 0.5, either part is enough. Any other result is mixed. The SNR axis and the pair of lower and upper layers are read the same way, as secondary.
- **What the paper says.** If the encoder carries the ranking gain, the LoRA mainly changes what the model represents of the audio. If the language model carries it, the LoRA mainly changes how the model reads out and answers. In both cases the part that moves the criterion is reported separately.
- **Limit.** The parts were trained together, so the fraction measures the contribution of each part within the full LoRA and not what it would learn alone.

### Execution order

All on Colab, in the order `cue-prov`, `cue-main`, `split-llm`, `split-audio`, `split-llm-upper` and `split-llm-lower`. About 7.5 T4 hours in total. While they run, the original versions of A and B and the joint contrast of advantages between banks (W2 of the review of draft v4) are computed on CPU.

### CPU answers to W2 and W3 of the review

Done on 22 October with `tools/intervention_analysis.py orig` and `w2`, on the features of `full-v1` and `prov-v3`. Exploratory. The fold heads reproduce the scores of `lib_v2.crossfit_scores` exactly (maximum difference 0), and chose C = 0.01 in all 5 folds.

**W3, the transfer drop of the same head.** Each fold head, trained only on duration variants, was applied unchanged to its held-out fold of the main bank and to the whole provenance bank. It scores 95.85 [95.09, 96.56] mean AUROC on its held-out fold and 91.16 [88.22, 93.82] on provenance. The gap of the same head is 4.69 [1.90, 7.73] points, with the recordings of the two banks resampled separately. Draft v4 gave 4.3, from a refit on the whole bank with fixed C compared against the cross-fitted heads. The drop holds when it is measured on the same head. With these heads, the in-bank advantage over fine-tuned L0 is 3.17 [2.35, 4.08], against 3.09 with all axes.

**W2, the joint contrast between banks.** It is the in-bank advantage over fine-tuned L0 on the main bank minus that on the provenance bank (+0.14 [−1.11, 1.48]), with all crops and the recordings of the two banks resampled separately.

| Heads of the main bank | Advantage on the main bank | Main minus provenance |
| --- | --- | --- |
| L2_cv with all axes | 3.09 [2.28, 3.98] | 2.95 [1.40, 4.57] |
| L2_cv with duration only | 3.17 [2.35, 4.08] | 3.03 [1.47, 4.69] |
| 182 excerpts per fold, mean of 10 draws | 1.53 [0.83, 2.25] | 1.39 [−0.11, 2.90] |

With full supervision the advantage on the main bank is clearly larger than on the provenance bank. With the supervision of the provenance heads the difference is 1.39 points and its interval includes zero. The interval for 182 excerpts covers only the evaluation sampling, because it averages the 10 draws within each replicate.

### Doubtful negatives of the provenance bank (W4 of the review)

Done on 22 October with `tools/screen_negatives.py` and `tools/intervention_analysis.py negatives`. Exploratory. An AudioSet tagger (AST, `MIT/ast-finetuned-audioset-10-10-0.4593`) scored the 1000 ms variant of the 228 clips. It is a filter and not a label. The speech clips have a median speech probability of 0.83 (0.69 at the 5th percentile). Of the 114 negatives, 18 reach 0.5, 26 reach 0.3 and 45 reach 0.1, and one sounds like a choir. The ranked list for listening is in `results/closing/negatives/screen.csv`.

The thresholds of 0.5 and 0.3 were fixed before looking at the effect. Each flagged recording is removed whole, with its speech clip, and no head is refitted.

| Negatives removed | Fine-tuned L0 | Fine-tuned L0 at 1000 ms | In-bank advantage | Gap of the same head |
| --- | --- | --- | --- | --- |
| None | 92.95 | 96.40 | +0.14 [−1.11, 1.48] | 4.69 [1.90, 7.73] |
| p ≥ 0.5 (18 recordings) | 94.70 | 98.86 | −0.45 [−1.64, 0.80] | 2.63 [0.15, 5.76] |
| p ≥ 0.3 (27 recordings) | 94.79 | 99.39 | −0.30 [−1.54, 1.05] | 2.30 [−0.15, 5.06] |

**Reading.**
- Without the doubtful negatives, the fine-tuned model at 1000 ms returns to its level on the main bank (99.66). Its drop at long durations on provenance came from negatives that probably contain speech.
- The in-bank advantage of the heads stays at zero.
- The transfer gap of the same head halves and its interval touches zero. Part of the drop of the transferred head came from contaminated labels and not from corpus cues.

### Exclusion of silent crops in both banks

Done on 22 October with `tools/intervention_analysis.py quiet`. The levels of the provenance crops come from `runs/prov-v3/banks/prov_bank_v3.metadata.csv.gz`. The exclusions apply only to evaluation, as in the reference.

| Exclusion on provenance | Crops removed | In-bank advantage | Matched reference | Decision |
| --- | --- | --- | --- | --- |
| None | 0 | +0.14 [−1.11, 1.48] | 1.53 | Shortcut |
| More than 20 dB below their clip | 59 | +0.16 [−0.97, 1.30] | 1.01 | Inconclusive |
| Below −30 dBFS | 157 | +0.04 [−1.16, 1.19] | 0.40 | Inconclusive |

With the exclusion applied in both banks the decisions do not change. The gap of the same head is 4.26 and 3.33.

### Check of the literature cited in the review

Done on 22 October against the arXiv pages. All the links exist and the review describes the papers correctly.
- **VocalGrad (arXiv:2610.23745).** It trains probes, a head-only LoRA and a full LoRA with the same data for each category (its section 5.4). Sharing the supervision does not set our work apart from it. The sentence in the related work and the one that closes that section were corrected.
- **Citation for the letter B.** The draft cited López et al., which studies the order and wording of the options but not a preference for a letter. The correct source is Zheng et al. (ICLR 2024), on the bias of language models towards option identifiers.
- **Glazer et al. (AAAI 2026).** Probes on the states of the Whisper decoder separate speech from non-speech at 100% in a set whose classes come from different corpora. It was in the bibliography and is now cited in the text.
- **SEAM (Interspeech 2026).** It deals with corpus and channel shortcuts in speaking-style classification. It is cited only if the corpus shortcut remains central. SEE and DEAF are peripheral.

### Blind listening to the provenance negatives

Done by the first author on 22 October with `tools/blind_listening.py`. It is a local page with the 114 negatives of 1000 ms, in random order, with neutral names and without the tagger's scores. The four categories were fixed before listening. The labels, joined with the clips and with the tagger's scores, are in `results/closing/negatives/listening.csv`.

| Label | Value in `listening.csv` | Clips | With p ≥ 0.5 in the tagger |
| --- | --- | --- | --- |
| Clear speech | `voz` | 25 | 10 |
| Possible or background speech | `voz_posible` | 13 | 6 |
| Singing, laughter or another vocal sound | `vocal_no_verbal` | 29 | 2 |
| No speech | `sin_voz` | 47 | 0 |

A third of the negatives (38 of 114) contain clear or possible speech. The tagger flags no clip without speech, but it misses 22 of those 38. Non-verbal vocal sounds count as non-speech, like the vocal classes of ESC-50 in the main bank.

**Provenance results without those recordings.** Each recording is removed whole, with its speech clip. The heads are not refitted, except in the last row.

| Recordings removed | Fine-tuned L0 at 1000 ms | In-bank advantage | Gap of the same head |
| --- | --- | --- | --- |
| None | 96.40 | +0.14 [−1.11, 1.48] | 4.69 [1.90, 7.73] |
| Clear speech (25) | 99.90 | −0.25 [−1.36, 0.88] | 1.95 [−0.43, 4.85] |
| Clear or possible speech (38) | 99.91 | −0.27 [−1.48, 1.02] | 2.32 [−0.35, 5.53] |
| All but those without speech (67) | 100.00 | −0.95 [−2.36, 0.46] | −0.04 [−2.44, 3.27] |
| Clear or possible speech (38), with the in-bank heads refitted without them | 99.91 | +0.61 [−0.37, 1.74] | 2.32 [−0.35, 5.53] |

**Reading.**
- The drop of the fine-tuned model at long durations on provenance was due to negatives with speech. Without them it is the same as on the main bank.
- The transfer drop of the same head is no longer detectable once the negatives with speech are removed, and it disappears when only the negatives without speech remain. It was largely label noise and not a corpus shortcut.
- The heads inside the provenance bank had been trained with a third of their negatives mislabelled. Refitted without them, they beat the fine-tuned model by +0.61 [−0.37, 1.74], within what the supervision of those heads predicts (see the reference with 122 excerpts below).
- The provenance control removes the confound between class and corpus, but it introduces label noise. The paper can no longer rest the corpus shortcut on the drop of the transferred head.

**Reference with the supervision of the clean heads.** Without the 38 recordings with speech, 76 remain, and each fold trains on about 61 recordings, that is, about 122 clips. With `closing_extra.matched_reference(42, 10, n=122)`, the advantage of L2_cv over fine-tuned L0 on the main bank, with 122 excerpts and durations only, is 0.98 without exclusion (from 0.49 to 1.51 across draws), 0.49 without the crops more than 20 dB below their clip and −0.07 without those below −30 dBFS. At 20 and 40 ms it is 3.35, 1.84 and 0.22.

**Joint reading.** With the same supervision and clean negatives, the in-bank advantage is 0.98 on the main bank and +0.61 [−0.37, 1.74] on provenance. The two agree. The 3.1-point advantage on the main bank comes from the amount of in-distribution supervision, and no difference between banks remains to attribute to a corpus shortcut. The corpus is still decodable at every depth, but that does not show that the speech heads use it.

### Result of I1, intervention on channel cues

Done on 22 October with `tools/intervention_analysis.py cue`, on `cue-main` (Kaggle) and `cue-prov` (Colab). The fold heads with the filter chose C = 0.01 in all 5 folds, as without the filter. Mean AUROC over the 8 durations, in points, with 2,000 replicates.

| Estimand | Without filter | With filter | Change | Reading under the rule |
| --- | --- | --- | --- | --- |
| A, advantage of L2_cv over fine-tuned L0 | 3.17 | 4.68 | +1.51 [0.89, 2.09] | The filter does not explain it |
| B, gap of the same head | 4.69 | 4.43 | −0.26 [−1.32, 0.88] | Inconclusive |
| B without the negatives with speech (outside the rule) | 2.32 | 2.26 | −0.06 [−1.22, 1.16] | |

**Controls.** With the filter, fine-tuned L0 drops by 1.97 [1.34, 2.54] on the main bank and by 1.93 on provenance, and the base model's native score drops by 3.30 and 2.88. The fine-tuned model stays below the 2-point threshold that the rule set to declare a change in difficulty, though very close to it. The decodability of the corpus barely drops, from 70.0 to 65.7 at 20 ms and from 97.0 to 95.4 at 1000 ms.

**Reading.** The VoxConverse band cut and the differences below 40 Hz do not explain the advantage of the in-bank heads. When they are removed the advantage grows, because the filter hurts the fine-tuned model, trained on full-band audio, and not the heads, which are trained on the filtered audio. The transfer gap does not change. Under the rule, the mechanism stays open. Together with the listening to the negatives and the matched reference, no evidence of a corpus shortcut remains. The advantage of the in-bank heads is explained by in-distribution supervision.

### Result of I2, LoRA restricted to part of the model

Done on 22 October with `tools/intervention_analysis.py split`, on `split-llm` and `split-audio` (Colab) and `split-llm-upper` and `split-llm-lower` (Kaggle). That is 300 base clips and 4,200 items. A first pass of the analysis had an error in the grouping by condition, and its numbers were discarded. The corrected analysis checks that the groups cover all items.

Means over the axis, in percent.

| Readout with L0 | AUROC duration | BA duration | AUROC SNR | BA SNR |
| --- | --- | --- | --- | --- |
| Base | 86.6 | 50.1 | 92.0 | 50.1 |
| Full LoRA | 91.9 | 83.7 | 98.8 | 96.7 |
| LoRA only in the encoder | 89.8 | 50.1 | 94.5 | 50.4 |
| LoRA only in the language model | 90.5 | 82.2 | 99.2 | 97.2 |
| Only layers 0 to 15 of the language model | 90.2 | 73.2 | 99.0 | 95.7 |
| Only layers 16 to 31 of the language model | 89.9 | 51.2 | 97.4 | 52.5 |

Fraction of the LoRA's gain that each variant recovers, with intervals from 2,000 replicates.

| Variant | R duration | C duration | R SNR | C SNR |
| --- | --- | --- | --- | --- |
| Encoder | 0.61 [0.50, 0.74] | 0.00 [0.00, 0.00] | 0.37 [0.18, 0.54] | 0.01 [0.00, 0.02] |
| Language model | 0.74 [0.62, 0.84] | 0.96 [0.94, 0.97] | 1.06 [1.00, 1.14] | 1.01 [1.00, 1.03] |
| Layers 0 to 15 | 0.68 [0.57, 0.78] | 0.69 [0.65, 0.73] | 1.02 [0.98, 1.10] | 0.98 [0.95, 1.01] |
| Layers 16 to 31 | 0.62 [0.54, 0.70] | 0.03 [0.02, 0.05] | 0.79 [0.69, 0.91] | 0.05 [0.03, 0.08] |

**Reading under the rule.**
- **Criterion.** On duration and on SNR the language model carries it, and within it, layers 0 to 15. The LoRA in the encoder does not move the criterion, and neither do layers 16 to 31.
- **Ranking, encoder against language model.** Mixed on both axes. The lower limit of the encoder on duration is 0.496, just below 0.5. Each part alone recovers much of the gain and the fractions add up to more than 1, so the effects overlap.
- **Ranking, lower against upper layers.** Either is enough on both axes.

**What the paper says.** The LoRA corrects the criterion inside the language model, in its lower layers. The ranking gain is spread out. The LoRA in the encoder alone recovers 61% of it on duration and 37% under noise, so the LoRA also changes what the encoder represents of the short crops. The parts were trained together, and the fractions measure contributions within the full LoRA.

### Provenance with clean negatives

Done on 22 October with `tools/clean_provenance.py` and with `ERCLM_PROV_RUN=prov-v3-clean` in `tools/closing_gpu.py prov`, `provcv` and `csweep`, which write their outputs under that run name. 76 recordings and 1,216 items remain. Heads trained inside the bank are refitted, and those that come from the training bank or from the main bank do not change.

- **The clean bank is easier for the native readouts.** The mean AUROC of the base model with L0 is 91.3 (88.7 with the whole bank) and that of the fine-tuned model 95.0 (92.9).
- **The transferred head loses its advantage.** With an AUROC of 93.7 it falls below the fine-tuned model, −1.31 [−2.96, 0.25], while on the main bank it beat it by 3.1. Its BA drops by 10 points because of the threshold.
- **The in-bank heads match or barely beat the fine-tuned model.** The base model gives +0.60 [−0.38, 1.71] and the fine-tuned model +1.10 [0.02, 2.29]. The layer profile is flat, from 95.4 to 96.2.
- **H3 on provenance.** The matched head stays below the fine-tuned model, −5.73 [−8.80, −2.94] AUROC.
- **Corpus.** It is still decodable, with AUROC of 72 to 79 at 20 ms and of 95 to 96 at 1000 ms.

## Review of draft v5 (23 October)

The review of draft v5 asks to narrow two causal explanations (N1 and N2), to complete the contrast of the clean bank against 122 excerpts (N3), to use the clean value of I1 (N4), to set I2 apart from Cho et al. (N5) and to complete the submission material (N6). As an optional GPU experiment it proposes the two combinations of I2 that are missing.

### Reading rule for I3, written on 23 October before extraction

**What changes.** The LoRA stays active in the encoder and in layers 0 to 15 of the language model (`split_audio_llm_lower`), or in the encoder and in layers 16 to 31 (`split_audio_llm_upper`). The same 300 clips and 14 conditions as in I2 are used. Together with the six earlier variants, all eight combinations of encoder, lower layers and upper layers are covered. Each half can then be removed from the full LoRA while the encoder stays adapted.

- **Measure.** The operating-point loss of a variant v, P(v), is the BA with the best threshold of each condition minus the native BA with L0, as a mean over the axis. The best-threshold BA is a diagnostic and not a readout. G = P(base) − P(LoRA) is the part of that loss that the full LoRA corrects. On the 300 clips it is 27.2 points on duration and 35.4 on SNR.
- **Estimands.** Δ_lower = P(encoder and upper layers) − P(LoRA) is what is lost by removing the lower layers. Δ_upper = P(encoder and lower layers) − P(LoRA) is what is lost by removing the upper layers. Intervals come from a bootstrap paired by recording with 2,000 replicates.
- **Reading, on duration (primary) and on SNR (secondary).** A half is necessary for the operating point if the lower limit of its Δ exceeds G/2, and is not necessary if the upper limit falls below G/4. Any other result is inconclusive. The change in AUROC and in native BA when each half is removed is reported as description.
- **What the paper says.** If the lower layers are necessary and the upper ones are not, the paper says that, with the encoder adapted, removing layers 0 to 15 from the LoRA brings back almost all of the base model's operating-point loss and removing layers 16 to 31 does not. The location survives the change of context. In any other case the paper limits the location to the restrictions tested. Neither reading turns BA into a pure measure of the criterion.
- **Code.** `lora_keep` accepts parts joined with "+" (`lib_v3.restrict_lora`). The analysis goes into `tools/intervention_analysis.py`.

### CPU analyses requested by the review of draft v5

Done on 23 October with `tools/intervention_analysis.py cleanjoint`, `splitextra` and `crossfilter`.

**N3, joint contrast on the clean bank.** Heads of the main bank with 122 excerpts per fold (mean of 10 draws within each replicate) against heads refitted inside the 76 clean recordings, with both banks resampled. AUROC in points.

| Durations | Main, head | Main, fine-tuned | Main, advantage | Provenance, head | Provenance, fine-tuned | Provenance, advantage | Difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 93.65 | 92.67 | 0.98 [0.27, 1.64] | 95.61 | 95.00 | 0.60 [−0.37, 1.74] | 0.38 [−1.01, 1.57] |
| 20 and 40 ms | 86.57 | 83.21 | 3.35 [1.74, 4.92] | 90.75 | 89.83 | 0.93 [−1.03, 3.13] | 2.43 [−0.22, 5.01] |

The single-draw advantage on the main bank goes from 0.49 to 1.51 over all durations and from 2.23 to 4.25 at 20 and 40 ms. Over all durations there is no detectable difference between banks. At 20 and 40 ms a residual of 2.4 points remains, with an interval that barely includes zero. There the fine-tuned model starts much higher on provenance (89.8 against 83.2).

**N1, I2 with best-threshold BA.** The BA with the best threshold of each condition is a diagnostic and not a readout. The operating-point loss is that BA minus the native BA with L0. On duration, in points.

| Variant | Native BA | Best-threshold BA | Operating-point loss | AUROC |
| --- | --- | --- | --- | --- |
| Base | 50.1 | 81.1 | 31.0 | 86.6 |
| Full LoRA | 83.7 | 87.5 | 3.8 | 91.9 |
| Encoder | 50.1 | 84.8 | 34.7 | 89.8 |
| Language model | 82.2 | 86.1 | 3.9 | 90.5 |
| Layers 0 to 15 | 73.2 | 86.0 | 12.8 | 90.2 |
| Layers 16 to 31 | 51.2 | 85.3 | 34.1 | 89.9 |

Layers 16 to 31 alone improve the ranking almost as much as layers 0 to 15 (best-threshold BA of 85.3 against 86.0), but they do not correct the operating point. On SNR the loss is 36.2 for the base model, 0.8 with the full LoRA, 1.6 with layers 0 to 15 and 42.3 with layers 16 to 31.

**Effect of the encoder conditional on the language model.** On duration, the encoder updates raise AUROC by 3.19 [2.36, 4.06] with the base language model and by 1.36 [0.84, 1.91] with the adapted one. On SNR they raise it by 2.52 [1.08, 3.95] with the base language model and not at all with the adapted one, −0.39 [−0.89, 0.01]. With the adapted language model, the adapted encoder adds native BA on duration, 1.50 [0.88, 2.12], and not on SNR.

**N4, cross-application of heads.** Heads of the main bank fitted on unfiltered or filtered audio, with the same folds, and scored on each version. Fine-tuned L0 comes from the scored version.

| Head fitted on | Scored on | Head AUROC | Fine-tuned L0 AUROC | Advantage |
| --- | --- | --- | --- | --- |
| Unfiltered | Unfiltered | 95.84 | 92.67 | 3.17 |
| Unfiltered | Filtered | 93.58 | 90.71 | 2.88 |
| Filtered | Unfiltered | 94.62 | 92.67 | 1.95 |
| Filtered | Filtered | 95.38 | 90.71 | 4.68 |

With the same heads, filtering at scoring time costs the head 2.3 points and the fine-tuned model 2.0, and the advantage barely changes. "The heads do not depend on the bands that the filter removes more than the fine-tuned model does."
[REFUTED 2026-10-23 by the follow-up review of draft v5 and the paired interval of this section. The head loses 2.26 [1.67, 2.84] and the fine-tuned model 1.97 [1.37, 2.55], and the advantage changes by −0.29 [−0.91, 0.35]. There is no detectable difference, but there is no bound that says "no more" either. Besides, the filter measures sensitivity to the filter and not the use of channel information.]
That the advantage grows with refitted heads reflects that they adapt to the filtered audio and the fine-tuned model does not.

### Result of I3

Done on 23 October with `tools/intervention_analysis.py i3`, on `split-audio-llm-lower` (Colab) and `split-audio-llm-upper` (Kaggle), both finished at 04:45 UTC in a single attempt. That is 300 clips and 4,200 items, with 2,000 replicates.

| Variant | AUROC dur. | BA dur. | Best threshold dur. | P dur. | AUROC SNR | BA SNR | Best threshold SNR | P SNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Encoder and layers 0 to 15 | 91.7 | 78.3 | 87.7 | 9.4 | 98.5 | 97.2 | 97.4 | 0.2 |
| Encoder and layers 16 to 31 | 91.5 | 52.4 | 87.2 | 34.8 | 97.1 | 58.9 | 95.7 | 36.7 |

**Reading under the rule.** G is 27.2 on duration and 35.4 on SNR.

| Axis | Half removed | Change in P | Reading |
| --- | --- | --- | --- |
| Duration | Layers 0 to 15 | +30.96 [28.46, 33.17] | Necessary |
| Duration | Layers 16 to 31 | +5.58 [4.42, 6.58] | Not necessary |
| SNR | Layers 0 to 15 | +35.94 [33.28, 38.56] | Necessary |
| SNR | Layers 16 to 31 | −0.56 [−1.44, 0.06] | Not necessary |

The upper limit for layers 16 to 31 on duration, 6.58, falls just below G/4 = 6.80. Removing a half barely changes AUROC on duration (−0.39 [−0.84, 0.04] and −0.16 [−0.38, 0.03]). Under noise, removing the lower layers costs 1.73 [0.69, 3.06].

**What the paper says.** With the encoder adapted, removing layers 0 to 15 from the LoRA brings back almost all of the base model's operating-point loss, and removing layers 16 to 31 does not. The location survives the change of context. As the rule says, BA is still not a pure measure of the criterion.

### Follow-up of the review of draft v5

The follow-up review of draft v5 considers N1, N2, N3 and N5 resolved, leaves N4 partly open because of a claim about the filter, and leaves N6 to the authors. The computations below are from 23 October with `tools/intervention_analysis.py crossfilter` and `i3`, with 2,000 replicates.

**Filter with the heads fixed.** With the heads fitted without the filter, filtering at scoring time changes the AUROC of the head by −2.26 [−2.84, −1.67], that of fine-tuned L0 by −1.97 [−2.55, −1.37] and the advantage by −0.29 [−0.91, 0.35]. With refitted heads the advantage changes by +1.51 [0.89, 2.15]. The paper gives the change in the advantage as it is, without claiming anything about dependence.

**I3 with G estimated in the same replicate.** This is a sensitivity analysis and does not replace the rule. Removing layers 16 to 31 falls below G/4 by 1.22 [0.18, 2.40] on duration and by 9.40 [8.81, 10.32] on SNR. Removing layers 0 to 15 exceeds G/2 by 17.35 [15.75, 18.69] and 18.25 [15.67, 20.53]. The labels of the rule hold. "Not necessary" means below the threshold, because layers 16 to 31 still correct 5.58 points on duration.

## Is the dominant failure the operating point? (23 October, afternoon)

**Motivation.** The review of draft v5 found the paper a diagnosis without a positive result of its own, H3 failed without a measured cause, and a residual remains at 20 and 40 ms. The data suggest a common pattern. The base model ranks well and answers badly (AUROC of 87 and 92 with a BA of 50). The frozen head with C = 0.0001 ranks under noise almost as well as the fine-tuned model (AUROC 99.1 against 99.3) and sits 1.86 BA points below it. The effect of the LoRA on the operating point lives in its lower layers. Three CPU analyses, with rules written before computing, test that reading.

### Rule for S, symmetric prompt

- **What.** On the 300 clips and 14 conditions of the prompt control, s_sim is the mean of s with "A) speech, B) non-speech" and with "A) non-speech, B) speech". In both formats A comes first and s is z(speech letter) − z(non-speech letter). A letter or position bias enters the two with opposite sign and cancels. It is a per-clip fix, with no labels or batches, that costs two passes. The secondary variant is the mean of the four formats.
- **Estimands.** On each axis, the BA of s_sim with threshold 0 minus the BA of L1 (batch calibration) on the original format. Also the BA of s_sim minus fine-tuned L0, the AUROC of s_sim minus the original AUROC, and the operating-point loss of s_sim (best-threshold BA minus native BA). Bootstrap paired by recording with 2,000 replicates.
- **Reading.** The symmetric prompt fixes the criterion without labels if BA(s_sim) − BA(L1) has a lower limit above −2 on both axes. It fixes it partially if it exceeds the original native BA by 15 points on both axes. Otherwise it does not fix it. The same is reported for the fine-tuned model.

### Rule for R, why H3 fails

- **R1, choice of C with a validation that simulates the distribution shift.** The training bank always carries noise and the duration axis of the test bank is clean. C is chosen from the pre-registered grid by a validation that leaves out one SNR level at a time, with the validation recordings separate from the training ones (4 folds by recording for each of the 6 levels). Secondary, train with SNR up to +10 dB and validate at +20 dB. The chosen C is refitted on the whole training bank.
- **R2, threshold of the head.** For the confirmatory head (C = 0.01) and the one from R1, the BA with the native threshold (0), with batch calibration on its score (as L1) and with a fixed threshold fitted on the training bank (as L1_tr), minus the BA of fine-tuned L0, per axis. AUROC and best-threshold BA are also reported.
- **Reading.** If some combination of a C chosen without looking at the test bank and a threshold set without test labels has a lower limit above −2 on both axes, a frozen readout with a calibrated criterion is not inferior to the fine-tuned model in the sense of H3. It is exploratory and does not change H3. If the BA gap falls by more than half when the threshold is calibrated, the failure of H3 is mainly one of operating point.

### Rule for N, the residual at 20 and 40 ms

- **Hypothesis.** The fine-tuned model on the main bank starts low at 20 and 40 ms (AUROC 83.2) because ESC-50 has vocal sounds that resemble speech in short crops. The heads inside the bank learn to reject them and the fine-tuned model does not.
- **What.** The joint contrast at 20 and 40 ms (heads of the main bank with 122 excerpts against the clean provenance bank) is repeated after removing from the main bank, in the evaluation only, the vocal classes of ESC-50. These are the non-verbal human ones (crying baby, sneezing, breathing, coughing, laughing, snoring) and the animal vocalisations (dog, rooster, pig, cow, frog, cat, hen, sheep, crow and birds).
- **Reading.** With the original residual of 2.43, the composition of the negatives explains it if without the vocal classes it falls to 1.22 or less, does not explain it if it stays at 1.82 or more, and explains it in part otherwise. Also reported are the AUROC of fine-tuned L0 at 20 and 40 ms without those classes and the advantage on the main bank. Secondary, also remove from the provenance bank the recordings whose negatives were heard as non-verbal vocal sound.

### Results of S, R and N

Done on 23 October with `tools/operating_point.py s`, `r` and `n` (outputs in `results/closing/S.md`, `R.md` and `N.md`), after the rules above were written.

**S, symmetric prompt. It does not fix the criterion.** In the base model, the BA of s_sim with threshold 0 is 55.75 over durations and 65.22 on SNR, against 77.96 and 81.78 for L1 on the original format. The differences are −22.21 [−24.75, −20.00] and −16.56 [−19.67, −13.22]. The gain over the original native BA is 5.67 and 15.17 points, so it does not even fix it partially on both axes. The mean does improve the ranking, +2.12 [1.05, 3.26] AUROC over durations and +4.21 [2.81, 5.83] on SNR. In the fine-tuned model almost nothing changes.

**Why it does not fix it (description after the rule).** With s_sim the mean score of non-speech stays fixed, between −1.63 and −1.89, at all durations and SNRs, and that of speech rises with the evidence, from −1.36 at 20 ms to −0.14 at 1000 ms. Besides the letter bias there is a constant shift towards "non-speech". The hypothesis that the silence of the container caused it is refuted, because non-speech does not move with duration and the shift persists on SNR, where the container is full of noise. A constant shift is exactly what a single threshold corrects, which explains the success of L1_tr.

**S with batch calibration (outside the rule).** L1 on s_sim gives 82.8 BA over durations and 89.7 on SNR, against 78.0 and 81.8 for L1 on the original format and 83.7 and 96.7 for fine-tuned L0 on the same 300 clips. It is a training-free combination that over durations comes within one point of the fine-tuned model. Since it was found by looking at these clips, it is tested in G1 with clips from other recordings.

**R, why H3 fails. The failure of the confirmatory head is one of ranking, not of threshold.**
- The validation that leaves out one SNR level and the one that validates at +20 dB both choose C = 0.01, the same as the confirmatory rule. The training bank always carries noise, so no internal validation sees the clean audio where the strong penalty performs well.
- With C = 0.01 the BA gap with fine-tuned L0 goes from −3.58 [−4.90, −2.32] to −3.23 with batch calibration and −3.79 with a fixed threshold over durations, and from −2.34 to −2.16 and −1.99 [−2.63, −1.40] on SNR. No combination meets the H3 margin, and the gap does not fall by half.
- With the best threshold in each condition the head reaches 81.33 against 84.16 for fine-tuned L0 over durations, for an AUROC of 87.53 against 92.7. The hypothesis that the dominant failure of the head was the operating point is refuted. That hypothesis holds for the native readout of the base model, not for the linear head.

**N, the residual at 20 and 40 ms. The vocal classes do not explain it.** Without them in the main bank the residual rises from 2.43 [−0.22, 5.01] to 3.05 [0.28, 5.57]. If the negatives heard as vocal sound are also removed from the provenance bank, it is 3.43 [0.17, 6.60]. The AUROC of fine-tuned L0 at 20 and 40 ms rises from 83.21 to 84.95, still far from the 89.83 of the provenance bank. The vocal-class hypothesis is refuted.

**Where the residual comes from (description after the rule).** At 20 and 40 ms, the advantage of the head with 122 excerpts over fine-tuned L0 per ESC-50 category is +11.5 in interior and domestic sounds (fine-tuned 78.5), +8.4 in non-verbal human sounds (77.0), +5.0 in natural soundscapes, +0.2 in animals and +0.6 in exterior sounds. The classes where the fine-tuned model fails most are sneezing, coughing, can opening, water drops, glass breaking, mouse click and drinking, impulsive or transient sounds that in 20 ms resemble a speech onset. The heads inside the bank see them and learn to reject them. The LoRA was trained with 32 ESC-50 sources, always in noise. It is a difference of content between the negatives of the two banks, not of channel, and it suggests training the LoRA with clean crops and transient negatives.

### Rule for G1, written on 23 October before extraction

- **What.** The base model with the prompt "A) non-speech, B) speech" on the 970 clips of the duration and SNR axes and on the training bank (`configs/prompt_ba_speech_second_full.yaml`). s_sim is the mean of s with that format and with the original.
- **Test data.** Only the clips from recordings that are not in the prompt control. These are 470 clips (135 speech clips from 60 recordings and 335 without speech), so no recording from the discovery set enters the test.
- **Readouts.** s_sim with batch calibration per condition (L1), and s_sim with a fixed threshold fitted on the training bank on s_sim (as L1_tr), which works clip by clip without a batch.
- **Primary.** BA of s_sim with L1 minus BA of fine-tuned L0 over durations. Not inferior if the lower limit exceeds −2 (the H3 margin). Bootstrap paired by recording with 2,000 replicates.
- **Secondary.** The same on SNR, the fixed-threshold version on both axes, and the gain BA(s_sim with L1) − BA(original with L1), positive if its lower limit exceeds 0.
- **What the paper says.** If the primary is not inferior, the paper reports a training-free recipe that matches the fine-tuned model on short clean speech, "tested on new recordings". If not, it reports that the estimate from the 300 clips did not replicate. In both cases SNR is reported.
  [REFUTED 2026-10-25 by the second reading of G1. They are not new. They are outside the prompt control but inside the main bank, already evaluated with the original format and with the fine-tuned model. G1 validates the recipe outside the discovery subset and is not a replication on a new bank.]

### Result of G1

Done on 25 October with `tools/operating_point.py g1` on the run `prompt-ba_speech_second-full` (Colab, finished on 23 October at 23:37 UTC in a single attempt). These are 470 clips from recordings that are not in the prompt control, 135 speech clips from 60 recordings and 335 without speech, with 2,000 replicates.

| Readout | BA duration | Minus fine-tuned L0 | BA SNR | Minus fine-tuned L0 |
| --- | --- | --- | --- | --- |
| Fine-tuned L0 | 84.55 | | 98.00 | |
| s_sim with L1 (primary) | 84.82 | +0.26 [−1.73, 1.99] | 91.60 | −6.39 [−7.68, −5.08] |
| s_sim with L1 on the full batch (sensitivity) | | +0.14 [−1.45, 1.89] | | −7.93 [−9.81, −6.14] |
| s_sim with fixed threshold | 82.76 | −1.80 [−3.83, 0.40] | 89.68 | −8.32 [−9.82, −6.82] |
| Original with L1 | 81.99 | | 86.20 | |

**Reading by the rule.** Over durations, the primary is not inferior to the fine-tuned model (lower limit −1.73 against the margin of −2). The gain over the original format with L1 is positive, +2.82 [0.93, 4.17] over durations and +5.41 [3.74, 7.50] on SNR, and so is the gain in AUROC, +2.03 [1.05, 3.07] and +3.83 [2.74, 5.13]. On SNR the recipe stays 6.4 points below the fine-tuned model. The fixed-threshold version does not meet the margin on either axis.

**Design flaw, declared.** The batch registered for L1 is the test clips, with 29% speech, and batch calibration assumes balanced batches. The sensitivity with the full batch of each condition (970 clips, balanced by design and without labels, with the threshold fixed across replicates) gives the same, +0.14 [−1.45, 1.89].

**Why it works over durations and not on SNR.** Averaging the two formats cancels the letter bias and averages the errors specific to each format. On the 300 control clips each format alone gave an AUROC of 86.6 and 86.7, and the mean 88.8. The calibration then corrects the constant shift towards "non-speech". "On short clean speech the fine-tuned model has no training advantage, because its bank always had noise. Under noise it does, and it ranks better (AUROC 99.3 against 96.3)."
[REFUTED 2026-10-25 by the second reading of G1 and `G1_filtro.md`. On the 470 clips the fine-tuned model also ranks better over durations, AUROC 94.03 against 91.41 (−2.62 [−3.79, −1.53]), and under noise 99.77 against 96.33. The recipe matches its native BA, not its ranking. That the cause is the noise of the training bank is a hypothesis (section "Consolidated second reading of G1").]

**Test after the rule. Conditioning on duration does not repair the fixed threshold.** Hypothesis. The fixed threshold loses against L1 because it was learned on a bank with noise. Per-duration thresholds fitted on the same bank go from −2.52 at 20 ms to −0.94 at 1000 ms, and lower the BA over durations to 74.72 (−9.84 [−11.96, −7.62] against the fine-tuned model). In the noisy bank, at short durations the optimal threshold drops to catch the speech of the −10 dB cells, and on clean audio that marks almost all non-speech as speech. The optimal single threshold on the test clips, which is only a diagnostic, is −1.29, against −1.46 from training. The hypothesis that conditioning on duration was enough is refuted. "What is missing is a threshold learned on clean audio."
[REFUTED 2026-10-25 by the second reading of G1. Discarding one solution does not prove another by elimination. A threshold learned on clean audio is a hypothesis, and the test is the first contrast of the clean control.]

**Common cause of three failures and an idea for improvement.** The training bank always has noise. "That explains the ranking gap of the frozen head on clean speech (R), the misplaced fixed threshold (this analysis) and the fine-tuned model's confusion with clean impulsive sounds (N)."
[REFUTED 2026-10-25 by the second reading of G1. It is a shared hypothesis, not a demonstrated cause. Each failure admits other explanations (amount of supervision, coverage of the negatives, mixture of durations) and is tested with its own contrast (section "Consolidated second reading of G1").] The idea is a training bank with clean cells and transient ESC-50 negatives. It would be enough to build it on CPU and extract it with the base model (about 40 minutes of T4) to refit the head and the threshold. Retraining the LoRA with that bank takes about 6 more hours. It is exploratory and does not change H3.

## Consolidated second reading of G1 (25 October)

A reviewer read G1. Each of the reviewer's claims was verified on the same 470 clips before it was adopted, with `tools/operating_point.py g1t` and three inline checks. The outputs are in `results/closing/G1_filtro.md` and `G1_transfer.md`. Everything here comes after the G1 rule and is exploratory.

**Verdict.** The reading holds almost entirely and is adopted. G1 can go at the centre of the paper if it says what it is compared against, what the recipe needs and where it stops being enough. There are three refinements. The recipe does not need to be calibrated on the test batch, but it does need an unlabelled batch per condition with a share of speech similar to that of the test. The compensation of errors between formats explains the AUROC gain over durations and not under noise, where the gain comes from putting speech in second place. Against a fine-tuned model calibrated the same way the recipe does not meet the margin.

| Claim | Label | Evidence |
| --- | --- | --- |
| The recipe reaches the BA of the fine-tuned model with its native decision over durations, +0.26 [−1.73, 1.99] | Observed evidence, registered rule | `G1.md` |
| Against fine-tuned L1 on the same clips (86.04) the recipe gives −1.22 [−2.97, 0.33]. It neither meets the −2 margin nor shows inferiority | Observed evidence, after the rule | `G1_filtro.md` |
| The fine-tuned model also ranks better on clean audio, AUROC 94.03 against 91.41 (−2.62 [−3.79, −1.53]). With the best test threshold, 89.00 against 85.96 | Observed evidence, after the rule | `G1_filtro.md` |
| Under noise the gap is one of ranking. AUROC 99.77 against 96.33, and the recipe with its best threshold (92.78) stays below the native BA of the fine-tuned model (98.00) | Observed evidence | `G1_filtro.md`, `G1.md` |
| The calibration transfers from separate unlabelled clips. With per-condition thresholds from the 300 control clips (50% speech), +0.17 [−1.43, 1.90] over durations and −7.86 on SNR | Observed evidence, after the rule | `G1_transfer.md` |
| It depends on the share of speech in the batch. With the 500 clips of the discovery recordings (70% speech), −3.10 [−4.84, −1.35] over durations and −12.2 on SNR. With batches of 150 clips the BA over durations is 81.2 with 10% speech, from 83.8 to 84.7 between 20 and 50%, 81.1 with 70% and 76.0 with 90% | Observed evidence, after the rule | `G1_transfer.md` |
| Over durations the mean beats both formats, +2.03 [1.03, 2.32] over the better one. Each format gives 89.38 and 89.14, with a within-class correlation of 0.37 to 0.56 | Observed evidence, after the rule | `G1_transfer.md` |
| Under noise `ba_speech_second` alone already gives 95.76 against 92.51 for the original, and the mean adds 0.57 [−0.06, 1.23] | Observed evidence, after the rule | `G1_transfer.md` |
| On the 300 control clips, under noise, the two formats with speech in second place give 96.41 and 96.09 with either letter, against 91.96 for the original | Observed evidence, after the rule | `G1_transfer.md` |
| The AUROC gain over durations comes from format-specific errors that compensate | Inference. A constant shift does not change the AUROC within a condition, and the two formats rank equally and correlate weakly | `G1_transfer.md` |
| *Hearing the Order* (Lin et al., arXiv 2510.00628, version 3 of 31 August 2026) studies the position bias of six audio models on multiple-choice questions and mitigates it with permutations and majority vote. It does not use Qwen2-Audio and does not address binary detection, calibration or fine-tuning | Observed evidence, reading of the paper | arXiv 2510.00628v3 |

**Why the calibration depends on the share of speech.** L1 puts the threshold at the logit of the batch's mean probability. That mean weights the mean of each class by its share, so with more speech the threshold rises and speech detections are lost. There is no shift between recordings. The per-class and per-condition means of s_sim differ by 0.13 or less between the 300 control clips and the test clips, and the threshold from the 300 clips agrees within 0.07 with that of the test clips reweighted to 50% speech. Over durations the BA hardly changes between 20 and 50% because near the optimum it is insensitive to the threshold. Under noise the range is narrower and the best batch has 30%, like the test. Own error, declared. An earlier check calibrated with the 500 clips at 70% speech and labelled them as the 300 balanced ones. That is where the claim that the recipe only worked on the test batch came from. It was never reported.

**Why the AUROC rises.** Cancelling a constant preference for B does not change the AUROC, as the reviewer points out. Over durations the two formats rank equally, correlate weakly within each class and their mean ranks better than either, which is what is expected if their errors compensate. Under noise the pattern is different. What ranks better is putting the speech option in second place, with either letter, and the mean with the original adds little. Over durations `ab_speech_second` drops to 82.84, so the effect of position changes with the condition. Why position matters under noise remains open.

**What is adopted from the review.**

- G1 at the centre, with its limit under noise and the separation between BA, AUROC and best threshold. The layer interventions become support.
- Three comparisons in the paper. Original format, symmetrisation and calibration separately. Recipe against the fine-tuned model on clean audio and under noise. BA against AUROC and the diagnostic best threshold.
- Say that G1 compares against the fine-tuned model with its native decision and also give the contrast against the calibrated fine-tuned model. Say that the recipe uses two passes and an unlabelled batch per condition, in which range of speech share it works, and that the fixed threshold does not meet the margin.
- Cite Lin et al. and Zhou et al. The defensible contribution is to measure together what these controls recover relative to fine-tuning and how that changes with the acoustic condition.
- The three claims that exceeded the evidence are marked as refuted in place.
- The clean control in two separate contrasts.

**What is refined.**

- The compensation of errors explains the AUROC gain over durations. Under noise the position of the speech option explains it.
- The contrast against the calibrated fine-tuned model, which the reviewer left without an interval, is inconclusive.
- The cheap check the reviewer asked for before spending GPU time is already done and changes the description of the recipe. It is not transductive, but it does not work with any prevalence either.

**Causes, alternatives and ideas for improvement.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| The recipe loses with batches of 10% or 70% speech | L1 absorbs the class share. Observed evidence | None plausible with equal per-class means | A calibration that estimates the share without labels, with the EM of Saerens et al. (2002) or with the crossing of two equal-weight components. The sweep from 10 to 90% has to stay flat. CPU, one hour, with a rule before computing |
| Under noise the recipe stays 6.4 points below | Ranking, AUROC −3.43 [−4.71, −2.38]. Observed evidence | That another format ranks better under noise | G2, the mean of the two formats with speech in second place. On the 300 clips it raises the AUROC under noise by 1.56 [0.88, 2.34] and the gap with the fine-tuned model falls from 7.0 to 4.7. One base-model extraction of about 2.5 h of T4, with the rule below |
| On clean audio the fine-tuned model ranks better than the recipe | Inference. The recipe only changes how the letter is read. With in-distribution supervision the heads on the base model's state beat the fine-tuned model, so the ranking is in the state and the letter does not express it | That paraphrases of the question recover part of it | Adding the other two orders did not help (88.76 against 88.77 on the 300 clips). Paraphrases remain, with one extraction per paraphrase |
| Misplaced fixed threshold, worse head on clean audio (R) and fine-tuned model confused by transients (N) | Shared hypothesis, the training bank always has noise | Amount of supervision, coverage of the negatives and mixture of durations | The clean control in two contrasts |

**Design of the clean control.** The first contrast compares the noisy bank against a mixture of clean and noisy audio, with the same sources, the same number of examples, the same labels and the same fit. It answers whether exposure to clean audio improves transfer to clean audio, and it is measured with the head and the fixed threshold of the base model. The second compares the original negatives against a larger coverage of ESC-50 transients, with the same acoustic mixture and the same size. It answers whether content coverage explains the errors with transients. Clean audio, negatives and quantity are not changed at once. Extracting each bank with the base model takes about 40 minutes of T4 according to the project's estimate, which is unverified. It needs a generator with clean cells and its rule before extraction. Retraining the LoRA, about 6 hours, is only worth it for a causal claim about its errors, and then the improvement on clean audio and any loss under noise are measured.

**How G1 stands in the paper.** Without training, averaging the score of the two letter assignments and calibrating it per condition with an unlabelled batch "reaches the BA of the fine-tuned model with its native decision on short clean speech", +0.26 [−1.73, 1.99], with batches of 20 to 50% speech.
[REFUTED 2026-10-26 by the per-duration table requested in the calibrated review of draft v6 (`perdur.md`). The +0.26 is the mean of the duration axis and hides a crossover. The recipe wins by 8.13 [2.54, 12.12] at 20 ms and loses by 6.93 [4.63, 9.70] at 1000 ms. It does not match the fine-tuned model on short speech. It beats it where the fine-tuned model's native threshold fails and loses where the fine-tuned model ranks better.] It does not match its ranking, −2.62 AUROC, nor the calibrated fine-tuned model, −1.22 [−2.97, 0.33]. Under noise it stays 6.4 points below, also with the best threshold. It is exploratory and does not change H3.

### Rule for G2, written on 25 October before extraction

- **What.** The base model with the prompt `ab_speech_second` ("B) non-speech" first and "A) speech" second) on the 970 clips of the duration and SNR axes and on the training bank (`configs/prompt_ab_speech_second_full.yaml`). s_seg is the mean of s with that format and with `ba_speech_second`. Both put speech in second place, with the letters swapped.
- **Why.** On the 300 control clips, under noise, the two formats with speech in second place rank better than the original, and their mean raises the AUROC by 1.56 [0.88, 2.34] over s_sim. Over durations it changes it by −0.40 [−1.11, 0.26]. It is a discovery estimate.
- **Test data.** The same 470 clips as in G1, 135 speech clips from 60 recordings and 335 without speech.
- **Readout.** s_seg with L1 per condition on the test clips, as in the G1 primary.
- **Primary.** BA of s_seg with L1 minus BA of s_sim with L1 on SNR. It improves if the lower limit exceeds 0. Bootstrap paired by recording with 2,000 replicates and seed 42.
- **Secondary.** Over durations, BA of s_seg with L1 minus fine-tuned L0, not inferior if the lower limit exceeds −2. AUROC of s_seg minus AUROC of s_sim on both axes. BA of s_seg with L1 minus fine-tuned L0 on SNR. As a sensitivity, the per-condition thresholds from the 300 balanced control clips, as in `g1t`.
- **What the paper says.** If the primary improves and duration remains not inferior, the recipe becomes s_seg and its gap under noise is reported. If not, s_sim stays and the paper reports that the estimate from the 300 clips did not replicate.
- **Declared flaw.** The 470 clips already evaluated s_sim, so this is the second recipe tested on them. s_seg was chosen with the 300 control clips, but after seeing on the 470 that the gap under noise was one of ranking and that `ba_speech_second` alone ranked better than the original.

### Rule for P, a calibration that does not depend on the share of speech, written on 25 October before computing

- **Why.** L1 puts the threshold at the logit of the batch's mean probability, which weights each class by its share. That is why the G1 recipe performs with batches of 20 to 50% speech and drops with 10, 70 or 90% (a range of 8.7 BA points over durations and 12.9 on SNR, `G1_transfer.md`). For BA the optimal threshold is the crossing of the two class densities with equal weights, which does not depend on the share. Without labels, those densities can be estimated with a two-component mixture with free weights.
- **P1, primary.** Within each condition, a mixture of two Gaussians with common variance is fitted by EM to the s_sim of the unlabelled batch, with free weights. The threshold is the crossing of the two components with equal weights, which with common variance is the midpoint of their means. The EM starts from the partition of the batch at the L1 threshold, runs up to 500 iterations and stops when the mean log-likelihood changes by less than 1e−9. If the weight of a component drops below 0.02 the mixture is declared failed in that batch, the L1 threshold is used and the number of times this happened is reported. The variance is common because with 15 clips of one class a separate variance is poorly estimated and a component can collapse.
- **P2, secondary, EM of Saerens et al. (2002).** The probabilities come from a Platt map of s_sim fitted with labels on the training bank. Within each condition the EM estimates the share of speech π̂ in the batch. The threshold is that of L1 on the batch reweighted to 50% with the probabilities adjusted by the EM, and with π̂ = 0.5 it coincides with L1. The EM alone is not enough, because with calibrated probabilities the maximum-BA threshold does not depend on the share and would repeat the fixed threshold, which did not meet the margin in G1.
- **Descriptive sensitivity.** P1 with free variances and the equal-weight crossing between the two means, or the midpoint if there is no crossing between them.
- **Data.** s_sim from G1 on its 470 test clips. The calibration batches are those of the `g1t` sweep, 150 clips from the discovery recordings with 10, 20, 30, 50, 70 and 90% speech and 200 draws per share, with the same seed. L1 is recomputed on the same batches.
- **Primary.** Over durations, the range of the P1 sweep, the highest mean BA minus the lowest across the six shares. It is flat if the range is 2 points or less.
- **Cost.** P1 has no cost if the lowest mean BA of its sweep is within 2 points of the mean BA of L1 with 50% speech on the same batches.
- **Secondary.** The range and the cost on SNR, and those of P2 on both axes. With a bootstrap paired by recording on the 470 clips, of 2,000 replicates and seed 42, P1 minus fine-tuned L0 calibrating with the test batch (re-estimated in each replicate), with the 300 control clips and with the 500 clips at 70% speech. Each is not inferior if the lower limit exceeds −2. Also P1 minus L1 with those three batches. The share of speech each method estimates is reported against the true one.
- **What the paper says.** If P1 is flat and without cost over durations, the recipe is calibrated with P1 and the paper says it works with batches of 10 to 90% speech. If it is flat with a cost, the recipe stays with L1 and its range of 20 to 50%, and P1 is mentioned as an alternative with its cost. If it is not flat, the paper keeps L1 with its range and says that the mixture did not remove the dependence. SNR is reported the same way, as a secondary.
- **Declared flaw.** The 470 clips and the sweep batches were already used with L1. The rule is written after seeing that L1 depends on the share and before computing P1 or P2. When G2 finishes, the same rule is applied to s_seg as a description.

### Result of P

Done on 25 October with `tools/operating_point.py p`, after the rule was written. The outputs are in `results/closing/P.md` and, for the causes, in `P_causas.md`. L1 on the same batches reproduces the `g1t` sweep figure by figure, so the draws are the same. Of the 16,800 mixtures in the sweep, 2 failed.

| Method | Range dur. | Lowest BA dur. | Range SNR | Lowest BA SNR |
| --- | --- | --- | --- | --- |
| L1 | 8.70 | 76.00 | 12.90 | 78.81 |
| P1, mixture with common variance (primary) | 1.08 | 82.66 | 0.38 | 90.67 |
| P1 with free variances (sensitivity) | 2.75 | 81.05 | 2.79 | 87.93 |
| P2, EM of Saerens et al. | 5.98 | 78.62 | 6.53 | 84.47 |

**Reading by the rule.** P1 is flat over durations, with a range of 1.08 points, and without cost, because its lowest BA (82.66 with 90% speech) is 1.95 points below L1 with 50% speech (84.61), just within the 2 points. On SNR it is flat (0.38) and without cost, and with any share it beats L1 with 50% (90.15). P2 is not flat on either axis. By the rule, the recipe is calibrated with P1 and the paper says it works with batches of 10 to 90% speech.

**Secondary.** P1 costs relative to L1 when the batch is favourable, and with P1 the recipe does not show non-inferiority against fine-tuned L0 with any batch.

| Calibration batch | P1 − fine-tuned L0 dur. | P1 − L1 dur. | P1 − fine-tuned L0 SNR | P1 − L1 SNR |
| --- | --- | --- | --- | --- |
| Test clips, 29% speech | −0.85 [−2.54, 0.53] | −1.11 [−2.52, 0.22] | −7.51 [−9.04, −5.76] | −1.11 [−1.76, −0.27] |
| 300 control clips, 50% | −1.19 [−2.62, 0.38] | −1.37 [−2.24, −0.49] | −6.81 [−8.34, −5.28] | +1.05 [−0.09, 2.29] |
| 500 discovery clips, 70% | −1.26 [−2.73, 0.28] | +1.85 [0.79, 2.91] | −6.90 [−8.45, −5.34] | +5.31 [3.23, 7.44] |

The share estimated by P1 grows with the true one but stays compressed towards the middle, 0.16, 0.22, 0.29, 0.44, 0.59 and 0.75 for 10, 20, 30, 50, 70 and 90% speech. P2 estimates 0.71 with 10% and 1.00 from 50% upwards.

**Design flaw, declared.** The cost was defined against L1 with its best batch and a tolerance of 2 points, and not against the non-inferiority margin relative to the fine-tuned model. That is why P1 passes the rule while costing about one point, and loses the non-inferiority that L1 had with batches of 20 to 50%. The rule is applied anyway, and the paper has to say both things.

**Why P1 stays flat.** Evidence. The mixture weights absorb the share of the batch and the midpoint of the means does not depend on the weights.

**Where the cost of P1 comes from.** Evidence, test after the rule. With thresholds from the 300 balanced clips, P1 loses against L1 mainly at 20 ms (63.3 against 69.4) and at 40 ms (72.3 against 77.5), and wins from 200 ms (92.1 against 89.7 at 500 ms). The midpoint of the true means of each class, a diagnostic with labels, gives 85.05 over durations against 84.72 for L1 and 83.36 for P1. The problem is not the midpoint rule but the fit of the mixture. At 20 and 40 ms the classes overlap and speech spreads 1.5 and 1.6 times more than non-speech. The speech component takes only the upper tail (estimated share of 0.16 and 0.25 in a balanced batch) and the threshold ends up high, −1.27 against −1.51 for the true midpoint at 20 ms. At −10 and −5 dB speech spreads less than non-speech (0.57 and 0.72 times) and P1 loses 3.3 and 2.0 points against L1. Free variances do not fix it.

**Why P2 fails.** Evidence, test after the rule. The Platt map from the training bank gives the non-speech of the test bank probabilities of 0.44 to 0.53, against 0.38 in training. With probabilities shifted towards speech, the EM of Saerens et al. feeds back on itself and drives the share to 1.

**Why the training bank does not calibrate the test bank.** Evidence on the shift, inference on its cause, test after the rule. It is the same thing that misplaces the fixed threshold of G1. "At 1000 ms the banks agree", with non-speech between −1.76 and −1.90 in training and between −1.86 and −1.92 on the SNR axis of the test bank, level by level.
[REFUTED 2026-10-25 by the second reading of P and `P_causas.md`. Non-speech agrees, within 0.12. Speech gives −0.22 to 0.06 in training and −0.36 to −0.21 in test, 0.05 to 0.27 lower in test.] At short durations they do not agree. In training, at 20 ms, non-speech gives −2.53 to −2.34 at all SNR levels, also at +20 dB, and speech −2.48 to −1.96. On the duration axis of the test bank, with the container almost silent, non-speech at 20 ms gives −1.68 and speech −1.36. These two means and those of the SNR axis are from the 970 clips of the bank, not only from the 470 test clips. Both classes sit 0.6 to 1.1 logit units higher without noise in the container, and a threshold or a map learned on the noisy bank ends up too low for short clean crops. The explanation not ruled out is the difference in non-speech sources (32 in training and 485 in test), although then the difference would also be expected at 1000 ms. The test that separates them is short crops of the same sources with and without noise in the container, the first contrast of the clean control.

**Causes, alternatives and ideas for improvement.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| P1 loses at 20 and 40 ms and at −10 and −5 dB | The mixture does not separate overlapping classes with different spreads. Evidence | Non-Gaussian class shapes | Estimate the share once per batch in the conditions where the classes separate (500 and 1000 ms) and fix it in the mixtures of the others, or use it to reweight L1 to 50%. In this bank each clip is in every condition, so the share is common. In real use it is not, and that has to be declared. CPU, minutes, with a rule before |
| P2 diverges | Platt map from the noisy bank, shifted on short clean crops. Evidence | None plausible | Test it again if the training bank has clean cells |
| Fixed threshold and Platt map misplaced | The noise in the container lowers the score of both classes on short crops. Inference | Different non-speech sources between banks | First contrast of the clean control. Registrable prediction, with clean cells the fixed threshold at 20 ms rises and the fixed-threshold version of the recipe gets closer to L1 |

## Second reading of P and framing of draft v6 (25 October)

A reviewer answered the question of how to frame draft v6. The authors adopted framing A on that basis. Each claim was verified against `P.md`, `P_causas.md` and a new test of the EM before it was adopted.

**Verdict.** It is adopted in full. It corrects two details of the container diagnosis and three expressions that exceeded what was measured, and it separates two results that the account mixed. Batch calibration shows non-inferiority over durations with favourable batches. The mixture reduces the sensitivity to the composition of the batch, costs performance and does not inherit that non-inferiority.

| Claim | Label | Evidence |
| --- | --- | --- |
| With separate calibration on the 300 clips, P1 loses against L1 by 6.0 points at 20 ms (63.32 against 69.36) and 5.2 at 40 ms (72.28 against 77.45) | Observed evidence | `P_causas.md`, table A |
| Part of the container diagnosis uses the 970 clips | Observed evidence, the code of `P_causas.md` uses the whole bank in section B | `P_causas.md` |
| At 1000 ms speech does not agree between banks, 0.05 to 0.27 lower in test | Observed evidence | `P_causas.md`, section B |
| The EM of Saerens et al. converges in 71 to 102 iterations to the boundary, π̂ = 1 − 10⁻⁶, starting from a mean probability of 0.57 to 0.66 | Observed evidence, test in this section in three conditions | inline output, not shipped |
| L1 with the 300 separate clips is not inferior over durations, +0.17 [−1.43, 1.90] | Observed evidence, after the G1 rule | `G1_transfer.md` |
| The means of the mixture are estimated together with the weights and can move with the batch | Inference. The estimated share follows the true one compressed towards the middle | `P.md` |
| To attribute the shift to the noise of the container, the crop and its level must be kept and only the noise outside it changed | Inference. In the training bank the noise covers the whole container, the crop included, so the current contrast mixes context and degradation | `src/erclm/bankgen.py`, `add_noise`, "White Gaussian noise over the whole container, scaled to the fragment's RMS" |

**What changes in the text.**

- "Does not depend on the share" becomes "showed little sensitivity to the share in the evaluated sweep", with the concrete range, 1.08 BA points over durations and 0.38 on SNR over six shares and batches of 150 clips.
- "Without cost" becomes "within the registered tolerance of two points". The worst mean is 1.95 points below L1 with 50%, and at 20 and 40 ms the loss is 6 and 5 points.
- The 20 to 50% range of L1 is observed stability of the BA and not demonstrated non-inferiority at each share. The percentiles of the sweep describe the variation between batches on a fixed test set.
- The container shift goes into the body in two sentences, as an observed shift of the scores whose explanation awaits an intervention. Saerens goes to the appendix, with "saturates" instead of "diverges".
- If G2 meets its rule, the new format replaces the symmetric one under L1, and its combination with P1 is reported separately, as a description.

**Framing A of draft v6.** Draft v6 keeps the title of draft v5. One main table of methods on the 470 clips with BA, AUROC and best-threshold BA per axis, and a two-panel figure with the share sweep. H1 to H4 in one compact paragraph, with the failure of H3 visible. One paragraph of diagnostics and limits instead of a section on causes. Probes, provenance, filter, penalty grid, supervision and layers move to the appendix.

**Idea for improvement it leaves.** The container contrast in its direct form. The same test crops at 20, 40 and 100 ms, at their level, with the container almost silent or with white noise only outside the crop, extracted with the base model. That is about 300 clips by 3 durations by 2 containers, about 1,800 items and 15 minutes of T4, plus the generator and its rule before extraction.

### Result of G2

Done on 25 October with `tools/operating_point.py g2` on the run `prompt-ab_speech_second-full` (Colab, finished at 23:27 UTC in a single attempt). These are the 470 clips of G1, with 2,000 replicates and seed 42. The output is in `results/closing/G2.md` and the causes in `G2_causas.md`.

| Contrast | Duration | SNR |
| --- | --- | --- |
| BA of s_seg with L1 minus s_sim with L1 (primary on SNR) | −1.46 [−2.35, −0.23] | +1.74 [0.95, 2.70] |
| BA of s_seg with L1 minus fine-tuned L0 | −1.20 [−2.97, 0.68] | −4.66 [−5.77, −3.48] |
| AUROC of s_seg minus s_sim | −0.70 [−1.24, −0.21] | +1.45 [0.95, 2.06] |
| BA of s_seg with control thresholds minus fine-tuned L0 (sensitivity) | −2.04 [−3.73, −0.41] | −5.65 [−7.18, −4.19] |

**Reading by the rule.** The primary improves, because its lower limit under noise exceeds 0. Over durations s_seg is not non-inferior to the fine-tuned model, because the lower limit of −2.97 is below −2. The rule requires both, so the recipe remains s_sim.

**Design flaw, declared.** The negative branch of the rule says that the paper reports that the estimate from the 300 clips did not replicate, and that does not describe this result. The estimate under noise did replicate (AUROC +1.45 [0.95, 2.06] against +1.56 on the 300 clips). What fails is duration, where on the 300 clips the change in AUROC was −0.40 [−1.11, 0.26] and here it is −0.70 [−1.24, −0.21]. The rule is applied anyway and the paper reports the trade-off.

**Why it wins under noise and loses over durations.** Evidence, test after the rule. Each format ranks better in a different regime. With speech second under the letter A (`ab_speech_second`) the AUROC is 60.7 at 20 ms, 70.8 at 40 ms and 97.7 at 1000 ms, and 96.9 on average under noise. With speech second under the letter B (`ba_speech_second`) it is 74.7, 83.5 and 91.7, and 95.8 under noise. The original format sits in the middle, with 71.6, 82.2 and 95.2, and 92.5 under noise. s_seg loses against s_sim only from 20 to 100 ms (74.7 against 75.2 at 20 ms and 82.4 against 84.7 at 40 ms) and wins or ties from 200 ms and at all SNR levels. In each mean the two formats correlate within class at 0.35 to 0.46, averaged over conditions, so both means combine different errors.

**Explanations not ruled out.** Hypothesis. The description of the non-speech option, "(music/noise/silence/animals)", comes first when speech goes second, and the order of those words could interact with a silent or a noisy container. It could also be a recency effect of the last option read. Neither was tested.

**Cheap test of an improvement, on the 300 discovery clips and not on the 470.** Averaging the three informative formats does not help. It gives −2.08 [−3.50, −0.79] against fine-tuned L0 over durations and −6.89 [−8.61, −5.06] under noise, against −0.96 and −7.00 for s_sim. The idea is discarded without spending the test clips.

**Ideas for improvement.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| The format that ranks best changes with the condition | Interaction between format and evidence regime. Observed evidence, mechanism open | Order of the words of the description, recency | Paraphrase the non-speech option without "noise" or "silence" and repeat the two formats with speech in second place on the 300 control clips, about 30 minutes of T4 per format on the duration axis |
| s_sim wins on short crops and s_seg on long ones and with noise | The same interaction. Observed evidence | None tested | Choose the format by duration, which is known, or by noise, which can be estimated without labels. Choosing it on the 470 would be a garden of forking paths, so a new bank of clips is needed, about 2.5 h of T4 per format |

**P on s_seg, a description foreseen in the rule for P.** `tools/operating_point.py p2`, output in `results/closing/P_seg.md`. On SNR P1 is flat, with a range of 0.96 points and the lowest BA at 92.13 against 92.48 for L1 with 50%. Over durations it is not, with a range of 4.28 points, from 76.1 to 80.4, below L1 at almost all shares, and on the test batch it loses 3.64 [1.97, 5.58] against L1. 36 of the 16,800 mixtures failed, against 2 with s_sim. This is what the cause of the cost of P1 predicts (evidence, test after the rule). The mixture needs classes that separate, and with speech second under the letter A the short crops separate worse (AUROC of 60.7 at 20 ms). So the stability of P1 with respect to the share depends on the score and is not a property of the method.

## Calibrated review of draft v6 (26 October)

The review read only the PDF of draft v6. The check of each point is in the section "Response to the calibrated review", below. Its only substantive improvement is to validate the fixed recipe where the class cannot be inferred from the corpus, and that is the rule for V.

### Rule for V, the fixed recipe on the clean provenance bank, written on 26 October before extraction

- **What.** The base model with the prompt `ba_speech_second` on provenance bank v3, 1,824 items from 114 recordings at the 8 durations without noise (`configs/prompt_ba_speech_second_prov.yaml`). s̄ is the mean of s with the original format (`prov-v3`, already extracted) and with `ba_speech_second`. The recipe does not change. It is s̄ with L1 per condition and with P1, with the same implementation in `tools/operating_point.py`.
- **Data.** The 76 recordings whose non-speech window was heard with no clear or possible speech (`prov-v3-clean`). Each recording contributes one speech window and one non-speech window, so the class cannot be inferred from the corpus or from the recording. Sensitivity with the 114.
- **Calibration separated by recording.** The recordings, sorted by name, are split into two halves with a permutation of seed 42. The per-condition thresholds of each half are computed without labels and applied to the other, so no recording is calibrated with its own clips. The batches have 50% speech by construction. In each bootstrap replicate the thresholds are re-estimated with the replicate's weights. Secondary, calibration on the whole evaluation batch, as in the G1 primary.
- **Primary.** BA of s̄ with cross-fitted L1 minus BA of fine-tuned L0, mean over the 8 durations. Bootstrap by recording without strata, because each recording has both classes, with 2,000 replicates and seed 42. Not inferior if the lower limit exceeds −2. Inferior by more than the margin if the upper limit is below −2. Otherwise inconclusive.
- **Secondary.** s̄ with cross-fitted L1 minus the fine-tuned model with cross-fitted L1. s̄ with cross-fitted P1 minus fine-tuned L0 and minus the fine-tuned model with cross-fitted L1. s̄ with cross-fitted L1 minus the original with cross-fitted L1. AUROC of s̄ minus that of the fine-tuned model and minus that of the original. s̄ with L1 on the whole batch minus fine-tuned L0. The same with the 114 recordings. Per duration, s̄ with cross-fitted L1 minus fine-tuned L0 and minus the fine-tuned model with cross-fitted L1, as a description.
- **What was known before.** On these 76 recordings the base model with the original format and the fine-tuned model are already known. With cross-fitted L1 the original stays at −2.88 [−5.59, −0.49] against fine-tuned L0, with a standard error of 1.33 (design calculation for this rule, with 1,000 replicates). `ba_speech_second` was never extracted on this bank.
- **Expected precision.** With a standard error close to 1.3, the primary comes out non-inferior only if the estimate exceeds about +0.6. If symmetrisation adds the same as in G1 (+2.8), the estimate would be close to 0 and the result would be inconclusive. The rule declares this before seeing the data.
- **What the paper says.** If non-inferior, that the fixed recipe recovers the native BA of the fine-tuned model also where speech and non-speech share a recording. If inconclusive, that this bank is not enough to decide, with the estimate and its interval, and whether the gain over the original has the same sign as in G1. If inferior, that the G1 result is limited to the main bank. In all three cases P1 and the calibrated fine-tuned model are reported.
- **Declared flaw.** The provenance bank was already used in other analyses and its 38 exclusions were decided after seeing results. It is a transfer check and not an independent replication.

### Response to the calibrated review

Done on 26 October. Each point was verified before it was adopted.

**Verdict.** It is adopted in full. Its question 3, the per-duration table, changed the reading of the central result. Its question 1 is the rule for V, with the extraction already queued. Han et al. is a direct antecedent of P1 and is cited.

**The axis mean hides a crossover.** Observed evidence, test after the rule (`tools/operating_point.py perdur`, output in `results/closing/perdur.md`, and diagnosis per condition in `perdur_causas.md`). On the 470 clips, with 2,000 replicates.

| Duration | Recipe L1 − fine-tuned L0 | Recipe L1 − fine-tuned L1 | Recipe with control thresholds − fine-tuned L0 | Recipe P1 − fine-tuned L0 |
| --- | --- | --- | --- | --- |
| 20 ms | +8.13 [2.54, 12.12] | +4.28 [−1.27, 8.29] | +9.07 [4.51, 13.80] | +3.18 [0.36, 5.65] |
| 40 ms | +3.93 [−0.81, 8.70] | +1.49 [−2.41, 5.76] | +5.83 [1.90, 9.85] | +1.40 [−2.68, 5.36] |
| 60 ms | +1.97 [−1.42, 5.70] | +1.02 [−1.84, 4.33] | +1.76 [−1.35, 4.90] | +0.58 [−3.00, 3.70] |
| 80 ms | +1.65 [−1.80, 5.56] | +0.70 [−2.46, 4.03] | +1.68 [−1.17, 4.69] | +1.83 [−1.87, 4.66] |
| 100 ms | +1.22 [−2.22, 4.41] | −0.93 [−4.12, 2.11] | +2.28 [−0.07, 4.95] | +2.43 [−0.33, 4.77] |
| 200 ms | −2.34 [−5.61, 1.33] | −3.59 [−6.48, −0.51] | −3.05 [−6.24, 0.19] | −1.35 [−4.89, 1.73] |
| 500 ms | −5.52 [−8.36, −3.26] | −5.74 [−8.49, −3.51] | −8.76 [−12.09, −5.52] | −5.95 [−8.95, −3.28] |
| 1000 ms | −6.93 [−9.70, −4.63] | −7.00 [−10.05, −4.47] | −7.43 [−10.00, −4.97] | −8.92 [−11.38, −5.99] |

**Why it wins on short crops.** Evidence on the threshold, inference on its cause. The fine-tuned model ranks better in all conditions, also at 20 ms (AUROC 80.2 against 75.2). But its native decision is very conservative on short clean crops. At 20 ms it detects 21.5% of the speech with 99.1% specificity, and its BA of 60.3 is far from the 73.1 of its best threshold. "On the whole test bank it detects 24.7% at 20 ms" [REFUTED 2026-11-03 by `runs/full-v1-lora/results/lora/condition_table.csv`. The hit rate of L0 at 20 ms is 0.24794, which rounds to 24.8%. Draft v10 repeats the 24.7.] The 47.0% at 40 ms remains unchecked. The recipe calibrates the threshold in each condition and reaches 68.4 at 20 ms with a worse ranking. In its training bank, with noise in the container, the same native threshold detects 68.8% of the 20 ms speech at 0 and at +20 dB, with 96.9% specificity. That comparison has a flaw, because those are the items it was trained on. That the fine-tuned model's threshold is misplaced by the clean container is the same hypothesis as for the base model, in the opposite direction, and the test is the container contrast extracted with the fine-tuned model as well.

**Why it loses from 200 ms.** Evidence. The fine-tuned model hardly errs on long crops (native BA of 99.1 at 1000 ms, AUROC 100.0), and the recipe stays at 92.2 with its best threshold, because its AUROC plateaus at 96.8 and its specificity at about 90%. It is the same ranking gap as under noise.

**What changes in the text.** The abstract and the conclusion stop saying that the recipe matches the fine-tuned model on short clean speech. They say that it matches its native BA in the mean over clean durations, winning at 20 and 40 ms, where the fine-tuned model's native threshold misses most of the speech, and losing from 200 ms. The table goes to the appendix and the crossover to the body. A new lesson is to report the per-condition contrasts next to the axis mean.

| Point of the review | What was done | Evidence |
| --- | --- | --- |
| Validate the recipe where the class cannot be inferred from the corpus | Rule for V registered before extraction, extraction on Colab | Rule for V, `configs/prompt_ba_speech_second_prov.yaml` |
| P1 has an antecedent in Han et al. | Verified. They fit a Gaussian mixture to unlabelled outputs, assign components to classes and drop the weights at inference, and they study batch imbalance. P1 is a one-dimensional case with common variance. It is cited in the method | arXiv:2205.10183v2 |
| What was known about the 470 clips when the rule was fixed | Both backbones had already scored them with the original format. The inverted prompt had not. This is stated in the method | Rule for G1 |
| Per-duration contrasts | Table above, to the appendix, and the crossover to the body | `perdur.md` |

**Idea for improvement.** If the native threshold of the fine-tuned model is what fails on short clean crops, calibrating it per condition fixes it only in part (fine-tuned L1 reaches 64.1 at 20 ms because L1 assumes a balanced batch and the test has 29% speech). The direct test is the container contrast with both backbones, about 30 more minutes of T4 for the fine-tuned model.

### Result of V

Done on 26 October with `tools/operating_point.py v` on the run `prompt-ba_speech_second-prov` (Colab, finished at 03:47 UTC in a single attempt), after the rule was written. Output in `results/closing/V.md` and causes in `V_causas.md`. These are the 76 clean recordings, with 2,000 replicates.

| Contrast | 76 clean (primary) | 114 (sensitivity) |
| --- | --- | --- |
| Recipe cross-fitted L1 − fine-tuned L0 | −1.64 [−4.19, 1.15] | −2.19 [−3.84, −0.22] |
| Recipe cross-fitted L1 − fine-tuned cross-fitted L1 | −2.63 [−5.18, −0.16] | −2.85 [−4.55, −0.99] |
| Recipe cross-fitted P1 − fine-tuned L0 | −2.38 [−5.26, −0.25] | −2.36 [−4.28, −0.60] |
| Recipe cross-fitted L1 − original cross-fitted L1 | +1.23 [−0.74, 3.54] | +2.58 [0.49, 4.06] |
| AUROC recipe − original | −0.91 [−2.43, 0.46] | −0.22 [−1.41, 0.84] |
| AUROC recipe − fine-tuned | −4.61 [−6.61, −2.80] | −4.52 [−6.22, −2.93] |
| Recipe cross-fitted L1 − fine-tuned L0, 20 ms | +15.13 [8.55, 21.71] | +13.60 [9.21, 19.30] |
| Recipe cross-fitted L1 − fine-tuned L0, 1000 ms | −13.16 [−19.74, −7.89] | −13.60 [−17.98, −8.77] |

**Reading by the rule.** Inconclusive, as the expected precision anticipated. The lower limit of −4.19 is below −2 and the upper limit of 1.15 above it. The BA gain over the original format has the same sign as in G1, +1.23 against +2.82, with an interval that includes zero. The paper says that this bank is not enough to decide non-inferiority, gives the estimate and says that the crossover by duration repeats.

**Why the ranking does not improve.** Evidence, test after the rule. On the provenance bank the inverted format ranks clearly worse than the original, AUROC of 84.57 against 91.30, and averaging with a weaker format adds nothing (90.39). On the main bank the two ranked equally (89.14 and 89.38). The gain from error compensation needs two formats that rank similarly, and that depends on the bank, probably on the type of negatives (VoxConverse windows without speech against ESC-50 clips). This last point is a hypothesis.

### The fine-tuned model calibrated in logit space

Test after the rules, 26 October (`tools/operating_point.py lora_cal`, output in `results/closing/lora_cal.md`). It is the cheapest test that could change the conclusion of the comparison against the calibrated fine-tuned model, and it changed it.

**Why L1 fails on the fine-tuned model.** Evidence. L1 averages probabilities, and the fine-tuned model has large logits that saturate them. At 20 ms, 90 to 95% of non-speech has a probability below 0.01 and the speech median is at −4.5 (main bank) or −4.8 (provenance bank), below the L1 threshold (−2.6 and −2.4). P1 does not separate the classes there either. The median of each condition's logits, the variant of L1 that equalises shares (`median_cond` in `lib_v2`, planned in [study_design.md](study_design.md) as a secondary variant), assumes a balanced batch and works in logit space.

| Balanced bank | Fine-tuned median − fine-tuned L0 | Best threshold − fine-tuned median | Recipe L1 − fine-tuned median | Recipe median − fine-tuned median | At 20 ms, recipe L1 − fine-tuned median |
| --- | --- | --- | --- | --- | --- |
| Main, 970 clips | +2.40 [1.39, 3.47] | +1.10 [0.82, 1.92] | −3.17 [−4.22, −2.21] | −3.87 [−4.86, −2.88] | +0.41 [−2.56, 2.93] |
| Provenance, 76 recordings | +7.57 [3.95, 10.12] | +0.74 [0.58, 2.96] | −9.13 [−11.27, −5.10] | −7.24 [−9.46, −3.70] | −10.53 [−14.47, −2.62] |

**Reading.** With a label-free calibration that works for the fine-tuned model, the fine-tuned model comes within one point of its best threshold and beats the recipe by more than the H3 margin on both banks. The recipe's advantage at 20 ms disappears on the main bank and reverses on the provenance bank. The tie of G1 with the native decision of the fine-tuned model is real, but it exists because that decision is misplaced on short clean crops. With both models calibrated, fine-tuning keeps an advantage in BA, not only in ranking. This comes after all the rules, and the median is valid only with balanced batches.

**What changes in the paper.** The central comparison has to be calibrated against calibrated. G1 is still reported as its rule says, against the native decision, and next to it goes the fine-tuned model calibrated with the median. A new lesson is to calibrate the fine-tuned model too, in logit space, before claiming that a training-free recipe matches it.

**Causes, alternatives and ideas for improvement.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| L1 does not correct the fine-tuned model | Saturated logits. Evidence | None plausible | A calibration in logits that does not assume balance, for example the mixture on clipped logits or a quantile threshold with the share estimated in the long conditions. CPU, minutes, with a rule before |
| The native threshold of the fine-tuned model fails on short clean crops | The LoRA was always trained with noise in the container. Inference | That the failure is specific to duration and not to the container | The container contrast with the fine-tuned model, about 15 minutes of T4 |
| The mean of formats does not improve the ranking on the provenance bank | The inverted format ranks worse with those negatives. Evidence for the symptom, hypothesis for the cause | Channel differences between banks | Choosing the format by its AUROC on an unlabelled calibration batch is not possible. It remains a limitation |

## Qwen3-Omni pilot (29 October)

The question was why the paper does not test Qwen3-Omni, and the authors chose a pilot before deciding on a full run. In the study under review, Qwen3-Omni without fine-tuning reaches 91.1% BA, against 93.3% for Qwen2-Audio with LoRA and 82.6% for the base model (README of `gbibbo/qwen-vad-lora`, generated answer, bank of the study under review). Per condition it wins under noise and loses on short crops. At 20 ms its BA is 0.50 because it detects no speech and gives no false alarms (table transcribed in `tests/fixtures/erc_lm_2027_SDT_duracion_SNR.csv`).

### Rule for the pilot, written on 29 October before extraction

- **What.** Qwen3-Omni-30B-A3B-Instruct (revision `26291f79`), thinker only, in NF4 with double quantisation and FP16 compute, batch 1, transformers 4.57.6, on Kaggle with 2 T4 (`configs/omni_pilot.yaml`, `src/erclm/omni.py`). The 100 clips of the Qwen2-Audio pilot (`select_pilot`, 50 per class, seed 42, test bank v2) at 20, 40 and 1000 ms. Three prompts read at the first token, `ab` (the confirmatory one), `ba_speech_second` (for s_sim) and the one of the study under review ("Does this audio contain human speech? Answer SPEECH or NONSPEECH."). s = z(speech) − z(non-speech) as in `lib_v2`.
- **Why.** To separate, in a newer and larger model, whether its failure on short crops is one of criterion or of representation, the same question the paper asks. If it is criterion, the recipe should extend to a second model and answer the single-model weakness that all the reviews point out.
- **Validity.** With the study prompt at 1000 ms the native BA has to reach 0.85. The study measured 0.971 by generating text. If it does not, the loading, the FP16 compute or the readout are wrong and the pilot is not read.
- **Feasibility.** With the measured time per item, the full run has to fit in 25 h of Kaggle. That is 970 clips by 14 duration and SNR conditions by the 2 formats of s_sim, 27,160 passes.
- **Registered reading.** For `ab` and s_sim at 20 and 40 ms, with 95% intervals by bootstrap over recordings stratified by class (2,000 replicates, seed 42). Representation if the upper limit of the AUROC is below 0.65. Criterion if the lower limit of the AUROC exceeds 0.60 and the best threshold gains at least 10 BA points over the native rule. Otherwise no conclusion. Evaluated in that order.
- **What is decided.** With validity and feasibility met, the full run is proposed, because either reading serves the paper. If feasibility fails, the subset of 300 clips of the prompt control is proposed.
- **Predictions.** If the failure is one of criterion, the AUROC at 20 ms resembles that of the Qwen2-Audio base model on the same clips and calibration per condition recovers most of it. If it is one of representation, the AUROC stays near 0.5. The second has a candidate mechanism. A 2 s container gives 26 audio tokens in Qwen3-Omni (about 77 ms each) and 50 in Qwen2-Audio (40 ms), so a 20 ms crop occupies a quarter of a token.
- **Reference on the same clips.** Qwen2-Audio with `ab` in `pilot-v1` (bank v2, NF4). The base model has AUROC 0.762, 0.854 and 0.911 at 20, 40 and 1000 ms, with native BA 0.50 in all three. The fine-tuned model has AUROC 0.830, 0.890 and 0.988, with native BA 0.64, 0.71 and 0.98. Verified in `runs/pilot-v1-base/` and `runs/pilot-v1-lora/`, `results/*/condition_table.csv`.
- **Declared flaws.** The study generated text and here the first token is read, so its BAs are not compared with these. With 100 clips the AUROC intervals span about ±0.09. The three durations and the reading thresholds were fixed after seeing the study's per-condition table.
- **Attempts.** `omni-pilot-v1` ran out of memory while quantising, before scoring any item. The rule does not change and the run continues as `omni-pilot-v2`, with the CUDA memory allocator in expandable segments.

### Result of the pilot

`omni-pilot-v2` ran on 29 October on Kaggle with 2 T4. Qwen3-Omni-30B-A3B-Instruct in NF4, thinker only, first-token readout, the 100 pilot clips of test bank v2. It loaded in 9 minutes and took 16.2 GB across the two GPUs, with nothing on the CPU. The FP16 logits are finite, the final state reproduces the logits and 2 s give 26 audio tokens (`checks_omni.json`).

**Registered reading** (`runs/omni-pilot-v2/results/omni/verdict.json`).

| Step | Value | Rule | Met |
| --- | --- | --- | --- |
| Validity, study prompt at 1000 ms | native BA 0.97 | at least 0.85 | yes |
| Feasibility | 4.17 s per item, 31.5 h for 27,160 passes | up to 25 h | no |
| `ab` and s_sim at 20 and 40 ms | AUROC of 0.58 to 0.67, with lower limits of 0.46 to 0.56 and upper limits of 0.68 to 0.77 | representation or criterion | no conclusion in all four cells |

The validity check reproduces the study, which measured 0.971 by generating text, so the loading, the FP16 compute and the first-token readout work. By the rule, the full run is not proposed and the subset of 300 control clips is proposed instead.

**Per condition, format `ab`.** Evidence.

| Condition | AUROC | Native BA | BA with L1 | BA with the best threshold |
| --- | --- | --- | --- | --- |
| 20 ms | 0.596 [0.475, 0.700] | 0.50 | 0.55 | 0.59 |
| 40 ms | 0.667 [0.550, 0.761] | 0.50 | 0.63 | 0.67 |
| 1000 ms | 0.986 [0.956, 1.000] | 0.94 | 0.94 | 0.98 |

At 20 and 40 ms it answers "non-speech" to everything, like the Qwen2-Audio base model and like Qwen3-Omni in the study. At 1000 ms, without fine-tuning, it decides well with its native threshold.

**Paired comparison with Qwen2-Audio on the same clips.** After the rule and descriptive (`tools/omni_paired.py`, bootstrap by recording stratified by class, 2,000 replicates, seed 42). Qwen2-Audio with `ab` comes from `pilot-v1` (NF4).

| Condition | AUROC Qwen3-Omni − base | AUROC Qwen3-Omni − fine-tuned | Best threshold, Qwen3-Omni − base |
| --- | --- | --- | --- |
| 20 ms | −0.166 [−0.319, −0.021] | −0.234 [−0.385, −0.094] | −0.11 [−0.21, 0.00] |
| 40 ms | −0.187 [−0.322, −0.060] | −0.223 [−0.357, −0.099] | −0.15 [−0.24, −0.06] |
| 1000 ms | +0.075 [0.030, 0.131] | −0.002 [−0.012, 0.003] | +0.13 [0.06, 0.18] |

**Reading.** On short crops Qwen3-Omni ranks worse than the Qwen2-Audio base model, which also fails with its native threshold. For Qwen2-Audio most of the failure is the threshold, and for Qwen3-Omni it is not, because not even the best threshold goes beyond 0.59 at 20 ms or 0.67 at 40 ms. The rule cannot call it representation, because with 100 clips the upper limit of the AUROC stays above 0.65. At 1000 ms Qwen3-Omni without fine-tuning matches the fine-tuned model in ranking and its native threshold is already well placed. The strong overall figure of Qwen3-Omni in the study holds for long clips and for noise, not for short ones.

"**The prompt matters in Qwen3-Omni.**" [REFUTED 2026-11-02 by the sweep of the crop position. The advantage is real with the crop on the seam and disappears at the centre of a block, where the two prompts rank equally at 20 and 40 ms (71.0 against 71.1). The block edge harms `ab` more.] With the study prompt it ranks better than with `ab`, +0.092 [0.018, 0.169] AUROC at 40 ms and +0.075 [−0.015, 0.160] at 20 ms. With that prompt, at 20 and 40 ms, the first token is "S" or "N" in only 23 and 16% of the clips. The model usually starts its answer with another word, and the first-token readout takes the difference between two tokens that are not its choice.

**Causes, alternatives and ideas for improvement.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| Qwen3-Omni ranks worse than Qwen2-Audio at 20 and 40 ms | Each audio token covers about 77 ms, against 40 ms in Qwen2-Audio, and a 20 ms crop is diluted in one token. Hypothesis | NF4 on a MoE of 30 billion parameters loses fine detail. The study ran in bf16 and also failed at 20 ms, but did not measure the ranking | Probes on the Qwen3-Omni encoder states in the crop region, as in H2a. If the encoder no longer separates, it is representation. About 40 minutes of T4 x2 plus the capture code |
| The study prompt ranks better | "The A/B format with letters is harder for Qwen3-Omni. Hypothesis" [REFUTED 2026-11-02 by the sweep. At the centre of a block both rank equally. What harms `ab` is the block edge] | The difference comes from reading a token the model did not choose | Read the probability of the full answer ("SPEECH" against "NONSPEECH") instead of the first token. Same cost as the pilot |
| 4.17 s per item | The per-expert loop of transformers 4.57 with bitsandbytes dequantisation in each expert. Inference | The split across two GPUs | A quantised format with fused MoE kernels (GPTQ or AWQ with vLLM). Exploratory and outside the design |

## Qwen3-Omni on the 300 control clips (30 October)

On 30 October the authors chose to run Qwen3-Omni on the subset that the pilot rule proposed, with the study prompt and with the probes.

While preparing the run a candidate mechanism appeared that the pilot did not consider. The Qwen3-Omni encoder cuts the mel frames into blocks of 100 (1 s), convolves each block separately and resets the position embedding in each one (`n_window` = 50 in the configuration of revision `26291f79`, `Qwen3OmniMoeAudioEncoder.forward` in transformers 4.57.6). A 2 s container gives two blocks of 13 tokens, and the centred crop falls exactly on the seam between them. A 20 ms crop is split between the last token of the first block, which covers 960 to 1000 ms, and the first token of the second. Qwen2-Audio processes the whole container without a seam. It is a hypothesis read from the code, with no data looking at it yet.

### Rule for Qwen3-Omni on the control clips, written on 30 October before extraction

- **What.** The same as the pilot (model, revision, thinker only, NF4, FP16 compute, batch 1, transformers 4.57.6, Kaggle with 2 T4, first-token readout) on the 300 clips of the prompt control (`select_pilot` with 150 per class and seed 42, duration and SNR axes) in the 14 conditions. These are the 4,200 Qwen2-Audio items of the prompt control and of `full-v1`. Four tracks, in this order. `ab`, `ba_speech_second`, `ab` with the audio rotated by 500 ms at the 8 durations (`ab_shift500`) and the study prompt (`hand_csl`). That is 15,000 passes, about 17 h over two Kaggle jobs (`configs/omni_control.yaml`).
- **States for the probes.** Each pass stores those of `lib_v2`. From the encoder before its projection (1280 dimensions, as in Qwen2-Audio), the mean over the container and the mean and the maximum over the crop tokens with one token of margin. The mean of the projection over those tokens, the final state and layers 12, 24 and 36 at the answer position (`omni.forward_scores`).
- **The rotated track.** Rotating by 500 ms moves the crop to the centre of the second block (tokens 18 to 20 at 20 ms, against 11 to 14 centred) and moves the last 500 ms of the container, almost silence on the duration axis, to the start. It does not change the audio of the crop or its level.
- **R1, the pilot reading with 300 clips.** The pilot rule unchanged, for `ab` and s_sim at 20 and 40 ms. Representation if the upper limit of the AUROC is below 0.65. Criterion if the lower limit exceeds 0.60 and the best threshold gains at least 10 BA points over the native rule. Otherwise no conclusion. 95% intervals by bootstrap over recordings stratified by class, 2,000 replicates, seed 42, in everything that follows.
- **R2, ranking against Qwen2-Audio.** AUROC of Qwen3-Omni with `ab` minus AUROC of the Qwen2-Audio base model with `ab` (`full-v1`) on the same items, paired by recording, in the mean of 20 and 40 ms and in the mean of each axis. Qwen3-Omni ranks worse on short crops if the upper limit in the mean of 20 and 40 ms is below 0. The same is reported against the fine-tuned model.
- **R3, the seam.** Δ is the AUROC of `ab_shift500` minus that of `ab` in the mean of 20 and 40 ms, paired by item. The gap is the difference of R2 with its sign changed. The seam explains most of the deficit if the lower limit of Δ exceeds 0 and Δ reaches half the gap. It explains a smaller part if the lower limit exceeds 0 and the upper limit is below half the gap. It does not explain it if the lower limit does not exceed 0 and the upper limit is below half the gap. Otherwise, no conclusion. If the gap is not positive, only Δ is reported. Secondary, Δ in BA with L1, Δ of the regional encoder probe and Δ at 1000 ms, where zero is expected.
- **R4, representation or readout.** Probes with cross-validation by recording within the 4,200 items, like L2_cv (`lib_v2.build_readouts` without a training bank, 5 folds, C by inner validation), with the same folds for both models. In the mean of 20 and 40 ms, (a) AUROC of the regional encoder probe (mean) of Qwen3-Omni minus that of the Qwen2-Audio base model, and (b) within Qwen3-Omni, AUROC of that probe minus that of s with `ab`. The failure originates in the representation if the upper limit of (a) is below 0. The encoder has the evidence and the readout does not use it if the lower limit of (a) exceeds −0.02 and that of (b) exceeds 0.05. Otherwise, no conclusion. Secondary, the same probe on the final state and on layers 12, 24 and 36, to locate where the evidence is lost.
- **Descriptives.** The recipe in Qwen3-Omni (s_sim with L1 minus the native rule of `ab`, minus fine-tuned L0 and minus the Qwen2-Audio recipe, per axis). The study prompt against `ab` per condition, with the proportion of first tokens that are a label.
- **Predictions.** If the seam is the cause, with `ab_shift500` the AUROC at 20 and 40 ms gets closer to that of the Qwen2-Audio base model and the regional encoder probe improves too. If the cause is that each token covers 80 ms, rotating changes nothing and the encoder probe stays below that of Qwen2-Audio. If the cause is in the language model (NF4 on the MoE or the A/B format), the encoder probe matches that of Qwen2-Audio and s does not.
- **What the paper says.** If the seam explains most of it, Qwen3-Omni is a second model whose failure on short crops has a design cause and a training-free fix, placing the crop inside a block. If R4 says representation, it is a second model where the threshold is not enough, unlike Qwen2-Audio. In both cases it goes in as an exploratory observation with a single backbone per model.
- **Declared flaws.** 100 of the 300 clips are those of the pilot (`select_pilot` takes the first ones of the same permutation), already seen at 20, 40 and 1000 ms. The R1 thresholds were fixed after seeing the study's table, as in the pilot. The seam hypothesis came from reading the code after the pilot and before seeing rotated data. The probes use labels of the test bank and are exploratory, like L2_cv. With 80 ms tokens the probe region of Qwen3-Omni covers more context than that of Qwen2-Audio (at 20 ms, 280 ms centred and 240 ms rotated, against 160 ms), which is part of the representation being compared. The 300 clips were already used to discover the Qwen2-Audio recipe, but never with Qwen3-Omni.

### Result of Qwen3-Omni on the 300 control clips

Done on 2 November with `tools/omni_control.py` on `omni-control-v1` (Kaggle, two jobs, finished on 31 October at 00:21 UTC), with Qwen2-Audio from `full-v1` and from `prompt-ba_speech_second`. Qwen3-Omni-30B-A3B-Instruct in NF4 (revision `26291f79`), thinker only, first-token readout, 300 control clips by 14 conditions of test bank v2, 2,000 replicates.

**Registered reading.**

| Rule | Measure | Value | Reading |
| --- | --- | --- | --- |
| R1 | AUROC of `ab` at 20 ms | 0.602 [0.539, 0.665] | no conclusion |
| R1 | AUROC of `ab` at 40 ms, with best threshold 0.66 against 0.50 for the native rule | 0.688 [0.628, 0.744] | criterion |
| R1 | s_sim at 20 and 40 ms | 0.595 and 0.688 | no conclusion and criterion |
| R2 | AUROC of Qwen3-Omni minus that of the Qwen2-Audio base model, mean of 20 and 40 ms | −0.128 [−0.206, −0.052] | ranks worse on short crops |
| R3 | AUROC with the crop rotated by 500 ms minus centred, mean of 20 and 40 ms | +0.066 [0.043, 0.091], against half the gap, 0.064 | the seam explains most of it |
| R4 (a) | AUROC of the regional encoder probe, Qwen3-Omni minus the Qwen2-Audio base model | +0.018 [−0.003, 0.038] | |
| R4 (b) | In Qwen3-Omni, the encoder probe minus s | +0.288 [0.230, 0.345] | readout, the encoder has the evidence and s does not use it |

**The seam.** Evidence. R3 is met by a very small margin, 0.066 against 0.064, and its interval runs from a third to seven tenths of the gap. The honest statement is that moving the crop off the seam recovers about half of the ranking deficit against the Qwen2-Audio base model. The effect falls with duration, as it should if the cause is that the crop falls only on the seam. The AUROC rises by 0.063 at 20 ms, 0.069 at 40, 0.066 at 60, 0.037 at 100, 0.007 at 200 and −0.002 at 1000 ms. With the rotated crop the native rule stays at 0.50 at 20 and 40 ms, because there Qwen3-Omni answers "non-speech" to everything, and rises from 0.517 to 0.627 at 60 ms and from 0.723 to 0.763 at 100 ms.

**The encoder has the evidence.** Evidence. The regional encoder probe reaches 0.933 at 20 and 40 ms, like that of Qwen2-Audio (0.915), and the probe on the final state 0.884, like Qwen2-Audio (0.885). The rotation does not change the encoder probe (0.920 against 0.907 at 20 ms, 0.946 against 0.948 at 40 ms). The seam does not erase the evidence in the encoder. It changes how much of it the untrained score expresses. "The compatible explanation (inference) is that the split crop lies in two tokens of different blocks, each at the edge of its convolution with zero padding and with the position reset." [REFUTED 2026-11-02 by R5c. Splitting the crop adds no harm (Δ at 1040 ms of +0.5 [−1.1, 2.2]). What harms is falling on the edge token of a block, split or whole.] A probe that averages the region and learns with labels of the bank compensates for it, and the frozen readout does not. The alternative not ruled out is that the region of the centred probe has four tokens and the rotated one three, and that this compensates for a loss. The sweep below separates splitting the crop from being at the edge of a block.

**The rest.** Evidence. Under noise and at 1000 ms Qwen3-Omni ranks better than the Qwen2-Audio base model (SNR +0.070 [0.049, 0.095], 1000 ms 0.995 against 0.949), as in the study. The recipe in Qwen3-Omni gains 8.0 [6.0, 10.1] BA points over its native rule over durations and 3.9 on SNR, and stays 5.0 [2.2, 7.8] below the fine-tuned model over durations. "The study prompt ranks better than `ab` in Qwen3-Omni (+0.053 [0.015, 0.092] at 20 and 40 ms) and worse in the Qwen2-Audio base model (−0.075 [−0.138, −0.014]), so the effect of the prompt on the ranking depends on the model." [REFUTED 2026-11-02 by the sweep. In Qwen3-Omni the advantage exists only with the crop on the seam. At the centre of the second block the two prompts rank equally.]

**For the study under review.** Its bank centres the crop in 2 s just like this one. Its 0.50 BA at 20 ms for Qwen3-Omni is the native rule answering "non-speech" to everything, and the seam does not change that at 20 and 40 ms. "At 60 and 100 ms the seam costs 11 and 4 points of native BA, and at all short durations it costs about half of the ranking difference with Qwen2-Audio. Its conclusion about short clips mixes a criterion failure of Qwen3-Omni with the position of the crop." [REFUTED 2026-11-02 by the sweep. Those costs are for `ab`. With the study prompt the seam moves the native BA by 2.7 [1.5, 4.0] points and does not clearly change the ranking at 20 and 40 ms (+1.2 [−2.5, 4.8]). At 60 and 100 ms it is not measured with that prompt. Its conclusion about short clips is mainly one of criterion.]

**Causes, alternatives and ideas.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| The seam costs half of the ranking deficit | "The crop split between two blocks convolved separately." [REFUTED 2026-11-02 by R5c. It is the block edge and not the split.] Evidence for the effect, inference for the mechanism | Being at the edge of a block, without being split. The absolute position in the container | The sweep below, about 6 h of Kaggle. In real use, align the analysis window with the 1 s blocks |
| The other half | 80 ms tokens, the language model readout in NF4 or the A/B format. Hypothesis | "The prompt, which in Qwen3-Omni ranks better with the question of the study" [REFUTED 2026-11-02 by the sweep. At the centre of a block the two prompts rank equally] | The sweep adds the study prompt with the rotated crop |
| The evidence is in the encoder and s does not use it | As in Qwen2-Audio, the failure is in the readout and not in the representation. Evidence | The probes use labels of the bank | For the paper, a second model with the same conclusion as the first |

## The study prompts on Qwen2-Audio (30 October)

The question was how this work treats the choice of prompt, which the study under review optimised with OPRO, and the authors decided to measure it on Qwen2-Audio. The paper fixes a single A/B prompt and separates ranking from threshold. What remained unknown is how much of what OPRO gained in the study is one and how much is the other.

### Rule for the study prompts, written on 30 October before extraction

- **What.** The two Qwen2-Audio backbones on the 300 clips of the prompt control and their 14 conditions, with three prompts of the study under review read at the first token (`configs/prompt_<p>.yaml`, `lib_v3.PROMPTS` and `lib_v2.PROMPTS`). `hand_csl` is the hand-written prompt used for all the models of the study, the same one used in Qwen3-Omni. `opro_llm_base` is the one OPRO-LLM chose for the base model (82.6% BA in the study). `opro_tmpl` is the one OPRO-Template chose for the fine-tuned model (93.3%). The texts come from `results/CONSOLIDATED_MATRIX_RESULTS.md` of `gbibbo/qwen-vad-lora`, which agrees with its OPRO analysis.
- **Estimands.** For each backbone and prompt p, paired with `ab` on the same 4,200 items and in the mean of each axis. G is the native BA of p minus that of `ab`, what choosing the prompt gives. O is the BA with the best threshold of p minus that of `ab`, the part that comes from the ranking. C = G − O is the part that comes from the operating point. Also reported are the AUROC of p minus that of `ab`, the BA with L1 of `ab` minus the native BA of p, the recipe (s_sim with L1) minus the native BA of p, and the BA with L1 of p minus that of `ab`. Bootstrap by recording stratified by class, 2,000 replicates, seed 42, with L1 re-estimated in each replicate (`tools/study_prompts.py`).
- **Reading.** Per backbone, prompt and axis. If the lower limit of G does not exceed 0.02, the prompt does not improve the native BA. If it does, the gain is mainly criterion if the lower limit of C/G exceeds 0.5, mainly ranking if its upper limit is below 0.5, and mixed otherwise. Calibration reaches what the prompt gives if the lower limit of the BA with L1 of `ab` minus the native BA of p exceeds −0.02. The primary readings are the base model with `opro_llm_base` and the fine-tuned model with `opro_tmpl`, the winner for each in the study.
- **Descriptive.** With `omni-control-v1`, the AUROC of the hand-written prompt minus that of `ab` in Qwen3-Omni and in the Qwen2-Audio base model on the same items. The proportion of first tokens that are a label, per prompt and condition.
- **Prediction.** The study describes that with OPRO the base model recovers sensitivity to speech at the price of more false alarms, a shift of the operating point. If so, in the base model the gain is mainly criterion and calibration reaches it. If it is mainly ranking, the prompt also changes how much evidence the score expresses, and the paper's reading that the base model fails because of the threshold needs that nuance.
- **Declared flaws.** The prompts were chosen in the study on its banks, not on the v2 banks. The study read generated text and here the first token is read, so its BAs are not compared with these. The 300 clips already served to discover the recipe. The prompt table in the paper of the study (`tables/Tab_PromptSummary.tex`) gives other texts for the hand-written prompt and for the OPRO-LLM prompt of the base model. Those of the consolidated matrix are used, which agree with its OPRO analysis and, for the hand-written prompt, with `07_qwen3_baseline/metrics.json`.

### Result of the study prompts

Done on 30 October with `tools/study_prompts.py` on `prompt-opro_llm_base`, `prompt-opro_tmpl` and `prompt-hand_csl` (Colab, one attempt each), with `ab` from `full-v1` and `ba_speech_second` from `prompt-ba_speech_second`. Qwen2-Audio-7B-Instruct in NF4 (revision `0a095220`), base model and retrained LoRA (checkpoint 400), first-token readout, 300 control clips by 14 conditions of test bank v2, 2,000 replicates.

**Registered reading.** In BA points, mean of the axis.

| Backbone and prompt | Axis | G | O | C/G | Reading | L1 of `ab` minus native BA of p |
| --- | --- | --- | --- | --- | --- | --- |
| base, `opro_llm_base` | duration | +18.0 [15.4, 20.6] | +0.5 [−1.1, 2.1] | 0.97 [0.89, 1.06] | mainly criterion | +9.8 [7.1, 12.8] |
| base, `opro_llm_base` | SNR | +34.4 [31.8, 37.0] | +5.8 [4.1, 7.3] | 0.83 [0.78, 0.88] | mainly criterion | −2.7 [−5.3, −0.1] |
| fine-tuned, `opro_tmpl` | duration | −19.4 [−22.0, −16.8] | +0.6 [−0.9, 1.8] | | does not improve | +20.5 [18.0, 23.0] |
| fine-tuned, `opro_tmpl` | SNR | −16.9 [−19.6, −14.3] | −1.1 [−1.9, −0.3] | | does not improve | +17.0 [14.4, 19.6] |

By the rule, in the base model what the prompt chosen by OPRO-LLM gives is mainly operating point on both axes. Calibration without search reaches it over durations and not under noise. In the fine-tuned model the OPRO-Template prompt worsens the native BA.

**Declared flaw, the first-token readout does not work for every prompt.** With `opro_llm_base` the first token of the base model is a label in only 49% of the duration items and 74% of the SNR items (15% at 20 ms). The model usually starts with "This", "non" or "Non", which are not in the label set, so the score compares two tokens the model does not choose. The study generated text and parsed "Non-speech." as non-speech. That is why our native rule gives speech for 60% of the non-speech duration clips, against 6.5% in the study's predictions (`02_base_opro_llm/predictions.csv`). The registered reading of `opro_llm_base` stands as an applied rule, but its G does not measure what the study measures. With the fine-tuned model and the hand-written prompt the first token is a label in 55 to 78% of the items, with the same problem.

**What does not depend on the flaw.** With `opro_tmpl` the first token is a label in 100% of the items for both backbones, and with `hand_csl` in the base model in 99% over durations and 87% on SNR.

| Base, against `ab` | Axis | G | O | C/G | AUROC | L1 of `ab` minus native BA of p | Recipe minus native BA of p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `opro_tmpl` | duration | +20.0 [17.5, 22.6] | +2.8 [0.9, 4.5] | 0.86 [0.78, 0.96] | +2.3 [0.4, 4.3] | +7.9 [5.2, 10.9] | +12.7 [9.7, 15.6] |
| `opro_tmpl` | SNR | +35.1 [32.3, 37.9] | +5.9 [4.0, 7.5] | 0.83 [0.78, 0.89] | +4.0 [2.7, 5.6] | −3.4 [−6.1, −0.9] | +4.6 [2.0, 7.2] |
| `hand_csl` | duration | +19.3 [17.4, 21.2] | −3.0 [−5.0, −1.0] | 1.15 [1.05, 1.26] | −4.3 [−7.2, −1.4] | +8.5 [6.5, 10.9] | +13.3 [11.1, 15.5] |
| `hand_csl` | SNR | +24.7 [21.4, 27.7] | −5.7 [−7.9, −3.4] | 1.23 [1.13, 1.35] | −6.6 [−9.1, −4.3] | +7.1 [4.1, 9.9] | +15.0 [12.3, 18.1] |

**Reading.** Evidence. In the base model, replacing the A/B format by one that asks for the word SPEECH or NONSPEECH mainly moves the operating point, 83 to 86% of the gain. The cause is the preference of the A/B format for the letter B, which leaves its native BA at 50 (prompt control). Any format without that preference gains almost everything through the threshold. The prompt also changes the ranking, in both directions. OPRO-Template ranks better under noise (+4.0 AUROC) and the hand-written prompt ranks worse (−4.3 and −6.6). "Calibrating `ab` without search beats OPRO-Template over durations and stays 3.4 points below under noise, where that prompt ranks better. The recipe beats both study prompts on both axes."

[REFUTED 2026-11-02 by the generation readout ("Result of the generation readout"). Read as the study read them, by generating, calibrating `ab` stays 6.9 [4.7, 9.4] points below OPRO-Template under noise in the base model. The recipe beats both prompts over durations and under noise ties with both, +1.0 [−1.1, 3.2] with OPRO-Template and +1.4 [−0.6, 3.6] with OPRO-LLM.]

**The fine-tuned model.** Evidence, with an untested mechanism. With OPRO-Template its ranking does not change (AUROC −0.3 over durations and −0.7 on SNR) and its native BA falls by 19.4 and 16.9 points. Its native rule gives speech for 70% of the non-speech duration clips, because the first token is "S" in 838 of 1,200. In the study, the fine-tuned model with that prompt wrote "NONSPEECH" for 94% of the non-speech duration clips and reached 93.3%. "There are two explanations not yet separated. 'S' may start another word, such as 'SILENCE', which the prompt names among the non-speech options (hypothesis). Or our LoRA retrained on the A/B format did not learn to answer with words like the study's (hypothesis)." The test that separates them is generating the text.

[REFUTED 2026-11-02 by the generation readout. Both hypotheses fall. The fine-tuned model writes SPEECH or NONSPEECH in all its answers except two "SNEEZE", and none is "SILENCE". The difference with the study is the repetition penalty of 1.1, which the first-token readout does not apply and which subtracts about 2 nats from "S" because the prompt contains that token. With the penalty applied, the first-step logits predict the first letter written in all items with "S" and "N" among the 20 highest logits.]

**Comparison with the hand-written prompt after the rule, descriptive.** OPRO-LLM against the hand-written prompt in the base model gives +9.7 [6.4, 13.2] native BA under noise, with +10.6 [8.1, 13.3] AUROC. It would seem that what OPRO gained over the hand-written prompt is ranking under noise, the opposite of the inference drawn on the morning of 30 October. But the OPRO-LLM side has the readout flaw, so this comparison neither refutes nor confirms that inference.

**Causes, alternatives and ideas.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| The first-token readout fails with free-answer prompts | The model starts with words that are not the label. Evidence (decoded tokens) | None for the base model. In the fine-tuned model, "S" may not be SPEECH | Read by generating up to about 10 tokens and parsing the text as the study did, or with the probability of the variants of each label. About 2.5 h of T4 for `opro_llm_base` on the base model and `opro_tmpl` on the fine-tuned model |
| In the base model the prompt mainly moves the threshold | The preference for the letter B leaves `ab` at 50 native BA. Evidence | That the word format changes the bias in another way | None new. It is what L1 and the recipe correct |
| The prompt changes the ranking by a few points, in both directions | The text of the question changes how much evidence the score expresses. Inference | That it is the bias of the token "S" or "N" and not the evidence | Average the scores of prompts that rank better under noise, as in G2, with a rule before and tested on the 470 clips |
| The fine-tuned model loses with OPRO-Template | "Its operating point remained tied to the format it was trained with. Inference" [REFUTED 2026-11-02 by the generation readout. The raw logit of "S" also rises with that prompt in the base model, and the study's repetition penalty compensates for it in both] | The first token "S" is not SPEECH. Differences with the adapter of the study | The generation above. For the paper, calibrate after any change of prompt |

## Generation readout of the study prompts (2 November)

On 2 November the authors decided to read the study prompts as the study read them, by generating the answer. With OPRO-LLM the base model usually starts with "This", "non" or "Non", and with OPRO-Template the fine-tuned model starts with "S" on 70% of the non-speech duration clips (section "Result of the study prompts").

### Rule for the generation readout, written on 2 November before extraction

- **What.** The two Qwen2-Audio backbones on the 300 control clips and their 14 conditions, with the three study prompts (`configs/gen_<p>.yaml`). Each item is generated with greedy decoding up to 16 new tokens and the text is parsed with the study's `normalize_to_binary`, ported unchanged into `src/erclm/genread.py`. UNKNOWN counts as non-speech, as in the study's predictions. The model's generation configuration is left as the study used it, which only turned sampling off, so its repetition penalty of 1.1 acts on the text. The same pass stores the first-token score of `lib_v2` and the 20 highest logits of the first step, without that penalty.
- **R-G1, the readout flaw.** The first-token readout misreads prompt p on an axis if the native rule (s > 0) and the parsed answer disagree on more than 10% of the items of that axis.
- **R-G2, calibrating against searching.** For the base model with `opro_llm_base` and the fine-tuned model with `opro_tmpl`, the winner for each in the study, calibration without search reaches the searched prompt if the lower limit of the BA with L1 of `ab` minus the BA of the parsed answer of p exceeds −0.02, per axis. The recipe minus that BA is reported too.
- **R-G3, the fine-tuned model's "S".** Among the non-speech clips where the fine-tuned model with `opro_tmpl` starts with "S", if at least half of the answers are parsed as non-speech, that "S" was not SPEECH and the drop of its native rule is a readout flaw. If not, the operating point of the fine-tuned model with that prompt is really shifted.
- **Descriptives.** The BA of the parsed answer and its difference with the native rule of `ab` (G_text), per backbone, prompt and axis. A ranking score built from the first tokens the model writes, using the parsed text and no labels (tokens whose answer is speech, or non-speech, in at least 90% of their occurrences, with 5 or more occurrences). Its AUROC and its BA with the best threshold against `ab`. The most frequent answers per class. The comparison with the study's predictions per axis.
- **Bootstrap.** By recording, stratified by class, 2,000 replicates, seed 42, a single draw for all readouts, with L1 re-estimated in each replicate (`tools/study_prompts_generation.py`).
- **Predictions.** If the flaw explains the difference with the study, R-G1 is met for `opro_llm_base` on the base model and the BA of the parsed answer gets close to that of the study over durations (82.5% there). If the fine-tuned model's "S" is "SILENCE" or another non-speech word, R-G3 shows it and the parsed BA of the fine-tuned model with `opro_tmpl` rises close to that of `ab`.
- **Declared flaws.** The study generated up to 128 tokens and here 16. A longer answer is cut and may be parsed differently. The ranking score uses tokens chosen by looking at the text of these same items, so it is descriptive. The comparisons with the study cross different banks and adapters.

**Amendment of 2 November, before the runs finished.** When checking the first chunk of `gen-opro_llm_base` (250 speech clips, base model) the answer "This is not human speech, it's a sound effect." appeared. `normalize_to_binary` reads it as speech, because it only recognises the negation "NOT SPEECH" with the two words together. The study did not use that parser for Qwen2-Audio but the `_parse_response` of its classifier (`src/qsm/models/qwen_audio.py`), which looks for a negation before SPEECH. Ported unchanged as `genread.parse_qwen2_audio`, it reproduces 21,339 of the 21,340 predictions of the study with OPRO-LLM on the base model and all 21,340 with OPRO-Template on the fine-tuned model. The one missing is "SPOKEN word.", which the study resolved with a fallback parser. `normalize_to_binary` agrees on 99.3% and 100%. The registered rule is applied anyway, with `normalize_to_binary`, and its flaw is declared. R-G1 to R-G3 are also computed with the study's parser for Qwen2-Audio as a secondary reading, decided after seeing that first chunk and before seeing any per-class result. The stored answers do not change.

### Result of the generation readout

Done on 2 November with `tools/study_prompts_generation.py --partial` on `gen-opro_llm_base` (base model, Colab) and `gen-opro_tmpl` (base and fine-tuned models, Kaggle, one attempt), with two code versions that generate identically. Qwen2-Audio-7B-Instruct in NF4 (revision `0a095220`), retrained LoRA (checkpoint 400), 300 control clips by 14 conditions of test bank v2, 2,000 replicates, seed 42. The two primary readings of the rule are complete. Still missing are the fine-tuned model with OPRO-LLM and both backbones with `hand_csl`, which are descriptive. The partial mode gives for each pair the same numbers as the full reading (`test_partial_reading_matches_the_full_one`). BA in points.

| Rule, with the registered parser | Base with OPRO-LLM | Fine-tuned with OPRO-Template |
| --- | --- | --- |
| R-G1. The first token misreads (disagrees on more than 10%) | Yes over durations (16.6%). No on SNR (8.8%) | Yes over durations (39.1%) and on SNR (15.5%) |
| R-G2. The BA with L1 of `ab` minus the BA of the text | +12.5 [9.9, 15.2] over durations, reaches it. −4.0 [−6.6, −1.3] on SNR, does not reach it | −1.7 [−2.9, −0.2] over durations, does not reach it. +2.2 [0.9, 3.7] on SNR, reaches it |
| The recipe minus the BA of the text | +17.3 [14.4, 20.0] and +3.9 [1.5, 6.4] | −0.8 [−2.0, 0.4] and +1.9 [0.9, 3.1] |
| R-G3. The fine-tuned model's "S" is not speech | | Yes. 84.0% of the 1,215 answers are parsed as non-speech |

With the study's parser for Qwen2-Audio, the secondary reading, the fine-tuned model does not change because the two parsers agree on all its answers. In the base model they disagree on 24.1% of the items, those that start with "This is not human speech", and the BA of the text goes from 65.5 to 77.0 [74.3, 79.5] over durations and from 85.8 to 88.3 [85.8, 90.8] on SNR. R-G1 is met on both axes (33.5 and 10.4% disagreement). R-G2 gives +1.0 [−1.2, 3.3] over durations, which reaches it, and −6.6 [−8.9, −4.3] on SNR, which does not. The recipe minus the text gives +5.8 [3.8, 7.7] and +1.4 [−0.6, 3.6].

**The base model with OPRO-LLM.** Evidence. Read as the study read it, it beats the native rule of `ab` through the criterion. Its gain over durations, +26.9 [24.3, 29.4], is all criterion, because the score built from the first tokens it writes ranks the same as `ab` (AUROC −1.6 [−3.7, 0.6]). That score uses tokens chosen on these same items, which favours the prompt, so the absence of a ranking gain is a conservative finding. Calibrating `ab` per condition reaches it over durations. Under noise it ranks better (+4.0 [2.7, 5.5] AUROC), the calibration stays 6.6 points below and the recipe ties. It is the pattern the first-token readout gave, now without its flaw. The registered parser costs it 11.5 points over durations because it reads "This is not human speech" as speech, and with that parser the calibration seems to win by 12.5 points. That advantage belongs to the parser and not to the model.

**The fine-tuned model with OPRO-Template.** Evidence. The drop of its native rule is the study's repetition penalty, which the first-token readout does not apply. In that prompt " NONSPEECH" is tokenised as " NON", "S", "PE", "ECH", so the token "S" is in the input and "N" is not. The penalty of 1.1 divides the positive logit of "S" and subtracts a median of 1.92 nats in the fine-tuned model and 1.70 in the base model. "The penalised first-step logits predict the first letter written in all items of both backbones that have 'S' and 'N' among their 20 highest logits, 99.6% in the fine-tuned model and all in the base model. In 1,224 items of the fine-tuned model and 1,523 of the base model the raw logit prefers 'S' and the model writes 'N', all because of the penalty. Without it the first letter would give the fine-tuned model 99.0% of speech detected and 42.7% of non-speech correct, and with it 89.0 and 90.8. In the base model it goes from 97.4 and 55.6 to 64.9 and 95.4." [REFUTED 2026-11-03 by `tools/repetition_penalty.py`, which replaces the ad hoc computation of 2 November. The penalised logits predict the first letter written in all items except two near ties per backbone, out of 4,198 items of the fine-tuned model and 4,200 of the base model with "S" and "N" or "NON" among the 20 highest. The items where the raw logit prefers "S" and the model writes "N" are 1,232 in the fine-tuned model and 1,532 in the base model. The fine-tuned model goes from 99.1 and 42.1 to 89.1 and 90.7, and the base model from 97.7 and 55.2 to 64.9 and 95.3.] Read this way, the fine-tuned model with OPRO-Template beats its native rule with `ab` over durations (+2.7 [1.1, 4.2]) and stays below it under noise (−2.2 [−3.7, −0.9]).

R-G3 is met and the drop of the native rule is a readout flaw. Declared flaw of the rule. It assumed that "S" started another non-speech word. The written answers that start with "S" are SPEECH except two "SNEEZE", and what turns "S" into "N" is the penalty. The conclusion of the rule holds with a different mechanism.

**The distance from the study.** With its parser the base model reaches 77.0 over durations and 88.3 on SNR, against 82.5 and 86.1 in the study. The fine-tuned model reaches 86.4 and 94.6, against 87.4 and 97.4 for the adapter of the study. The prediction for the base model is half met. R-G1 is met and the BA rises from 68.1 with the first token to 77.0, but 5.5 points remain over durations. They lie between 20 and 500 ms and are mostly false alarms, 6 to 31% per duration against 1 to 8% in the study. At 1000 ms our BA is higher (90.3 against 81.4). Hypothesis. The clips are different, only 14 of our 4,200 items are in the study's test set, and our short-duration negatives seem harder. The prediction for the fine-tuned model is met. The BA of the text rises close to that of `ab`, although because of the penalty and not because of another word.

**Flaw of a descriptive.** The ranking score of the generation does not exist for OPRO-Template. The raw "S" starts both answers and no token passes the 90% filter for speech, so its AUROC, O and C are not interpreted. The ranking of that prompt is measured by the first-token score (previous section).

**Causes, alternatives and ideas.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| The fine-tuned model with OPRO-Template recovers its BA when generating | The repetition penalty subtracts about 2 nats from "S", which the prompt contains. Evidence | None for our adapter. For the study's adapter it is an inference, with the same prompt, tokeniser and decoding | Declare it in the paper of the study under review. Decoding is part of the operating point of a prompt. No GPU needed |
| The raw logit of "S" rises with OPRO-Template in both backbones | Bias towards a token that appears in the prompt. Hypothesis | That the list of non-speech sounds in the prompt moves it | Read the first token with a variant that has no "S" token in the prompt, for example "NON-SPEECH", with a rule before. About 30 min of T4 per backbone |
| OPRO-LLM beats `ab` through the criterion over durations and ranks better under noise | The same as with the first token. Evidence | The ranking score uses tokens chosen on these items | None new. It is what L1 and the recipe correct |
| The base model stays 5.5 points below the study over durations, through false alarms from 20 to 500 ms | Different clips and banks. Hypothesis | The precision of the study's model, a different construction of the crop | Read the study's test items with this reader if their audio is available. Otherwise, compare the classes of the negatives that fail with those of the study, on CPU |

### Descriptives of the generation readout (3 November)

Done on 3 November with the full `tools/study_prompts_generation.py` and with `tools/repetition_penalty.py`. It adds `gen-hand_csl` (base and fine-tuned models, Kaggle, one attempt) and the fine-tuned model of `gen-opro_llm_base`, which moved to Kaggle (`gen-opro_llm_base-3`) after 32 Colab disconnections without saving a single chunk. Same model, bank and bootstrap as the primary readings. The BA of the text uses the study's parser for Qwen2-Audio.

| Backbone and prompt | BA of the text, duration | BA of the text, SNR | L1 of `ab` minus the text, duration | L1 of `ab` minus the text, SNR |
| --- | --- | --- | --- | --- |
| Base, hand-written prompt | 55.7 [54.4, 57.0] | 62.4 [59.7, 64.9] | +22.3 [20.1, 24.8] | +19.4 [16.2, 22.6] |
| Base, OPRO-LLM | 77.0 [74.3, 79.5] | 88.3 [85.8, 90.8] | +1.0 [−1.2, 3.3] | −6.6 [−8.9, −4.3] |
| Base, OPRO-Template | 73.8 [71.9, 75.8] | 88.7 [86.1, 91.1] | +4.1 [2.2, 6.3] | −6.9 [−9.4, −4.7] |
| Fine-tuned, hand-written prompt | 79.6 [77.5, 81.7] | 94.1 [91.9, 96.2] | +5.1 [3.8, 6.5] | +2.7 [1.3, 4.3] |
| Fine-tuned, OPRO-LLM | 84.2 [81.7, 86.5] | 94.7 [92.5, 96.6] | +0.6 [−0.7, 2.0] | +2.1 [0.8, 3.6] |
| Fine-tuned, OPRO-Template | 86.4 [84.1, 88.6] | 94.6 [92.3, 96.6] | −1.7 [−2.9, −0.2] | +2.2 [0.9, 3.7] |

**The penalty also acts with the hand-written prompt, against it.** Evidence. The hand-written prompt ends in "Answer SPEECH or NONSPEECH.", with the same " NON", "S", "PE", "ECH", so "S" is in the input. The penalty subtracts a median of 1.70 nats in the base model and 1.76 in the fine-tuned model. The penalised logits predict the first letter written in all items except two near ties in the base model. With that prompt both backbones were already conservative in raw form and the penalty pushes them further towards "non-speech". The base model goes from 48.2% of speech detected and 95.6% of non-speech correct to 18.7 and 98.5, and the fine-tuned model from 84.7 and 93.6 to 76.0 and 95.7. The BA of the base model falls by 14.1 [12.5, 15.7] points over durations and 12.4 [10.2, 14.6] under noise, and that of the fine-tuned model by 5.7 [4.5, 6.9] and 0.2 [−0.8, 1.3].

**The advantage of OPRO-Template over the hand-written prompt is mostly decoding.** Evidence on this bank. BA of the first letter, with and without the penalty.

| OPRO-Template minus hand-written prompt | Duration | SNR |
| --- | --- | --- |
| Base, without penalty | +0.2 [−2.6, 3.3] | +10.2 [6.9, 13.8] |
| Base, with penalty, as the study generated | +18.1 [16.3, 20.0] | +26.2 [23.6, 28.9] |
| Fine-tuned, without penalty | −21.1 [−23.6, −18.8] | −15.1 [−17.3, −12.8] |
| Fine-tuned, with penalty | +6.8 [5.1, 8.5] | +0.4 [−0.8, 1.8] |

The same penalty does opposite things depending on where the raw criterion was. With OPRO-Template the raw criterion is liberal (97.7% of speech detected and 55.2% of non-speech correct in the base model, 99.1 and 42.1 in the fine-tuned model), and shifting it towards "non-speech" adds 3.7 [1.0, 6.6] and 22.2 [19.8, 24.6] points over durations. With the hand-written prompt the raw criterion is already conservative and shifting it subtracts. In the base model, over durations, all the advantage of OPRO-Template over the hand-written prompt is decoding. Under noise 10.2 of the 26.2 points remain without the penalty, and that remainder is due to the prompt. In the fine-tuned model, without the penalty, the hand-written prompt wins by more than 15 points on both axes. Inference for the study. It generated this way for all its Qwen2-Audio prompts that name NONSPEECH, so its differences between prompts mix the text with a shift of "S" of 1.7 to 1.9 nats in median. Declared flaw. The BA without the penalty is that of the raw first letter and not that of a text generated without the penalty. In these two prompts the first letter decides the parsed label except in a few answers such as "SNEEZE", "SPOKEN word…" and "Bark.", but no text was generated without the penalty.

**The fine-tuned model with OPRO-LLM.** Evidence. It answers with words ("SPEECH", "NON-SPEECH", "non-speech") and sometimes with the base model's sentence, "This is not human speech…", which the registered parser reads as speech (81.0 over durations with that parser). With the study's parser it reaches 84.2 over durations and 94.7 under noise, the same as its native rule with `ab` over durations (+0.5 [−1.1, 1.9]) and 2.1 [0.8, 3.5] below under noise. Calibrating `ab` reaches it over durations and beats it under noise. With the fine-tuned model the three prompts lie between 79.6 and 86.4 over durations and between 94.1 and 94.7 under noise, and most of what the hand-written prompt loses over durations is the penalty (5.7 points). Its first token is not a label letter, so the penalty was not measured with this prompt.

**Causes, alternatives and ideas.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| In the base model the advantage of OPRO-Template over the hand-written prompt is decoding over durations | The penalty subtracts about 2 nats from "S" in both prompts and the raw criterion of each one sits in a different place. Evidence | That generating without the penalty changes something after the first letter | Generate both prompts on both backbones without the penalty, about 4 h of T4, with a rule before. For the paper of the study under review, decoding is part of the operating point of each prompt and is declared |
| The hand-written prompt is the most harmed | Its raw criterion is already conservative. Evidence | None | Compare prompts by ranking or with calibration per condition, which remove any fixed shift. Already done with the first token in "Result of the study prompts" |

## Sweep of the crop position in Qwen3-Omni (2 November)

R3 found that moving the crop off the seam recovers about half of the ranking deficit. This is the test that had been planned for the case in which R3 confirmed it.

### Rule for the sweep, written on 2 November before extraction

- **What.** The same as `omni-control-v1` (model, revision, NF4, unchanged `omni.py` code, Kaggle with 2 T4) on the same 300 clips at 20, 40 and 60 ms, with `ab` and the audio rotated to put the centre of the crop at 500 ms (centre of the first block), 750 ms (first block), 1040 ms (whole within the first token of the second block, not split), 1250 ms and 1750 ms (second block). The positions 1000 ms (the seam) and 1500 ms (centre of the second block) come from `omni-control-v1`. It adds the study prompt with the centre at 1500 ms at 20 and 40 ms. That is 5,100 passes, about 6 h (`configs/omni_seam.yaml`).
- **Measure.** Δ(x) is the AUROC of `ab` with the centre at x minus the AUROC with the centre on the seam, in the mean of 20, 40 and 60 ms, paired by item. Bootstrap by recording stratified by class, 2,000 replicates, seed 42 (`tools/omni_seam.py`).
- **R5a.** The seam harms at all tested positions if the lower limit of Δ exceeds 0 at 500, 750, 1250, 1500 and 1750 ms.
- **R5b.** What matters is the position within the block and not within the container if the interval of the AUROC at 500 ms minus that at 1500 ms (the two block centres) contains 0.
- **R5c.** Splitting the crop between blocks explains most of it if Δ(1040) has a lower limit above 0 and reaches half of Δ(1500). Being at the edge of a block also harms if the lower limit of the AUROC at 1500 ms minus that at 1040 ms exceeds 0. Both readings can hold at once.
- **Descriptives.** The AUROC of the study prompt with the centre at 1500 ms minus that of `ab` at the same position, and the gap that remains with the Qwen2-Audio base model. The encoder probes per position. The native BA and the BA with L1 per position.
- **Predictions.** If the cause is splitting the crop, Δ(1040) resembles Δ(1500) and all positions off the seam gain the same, about 0.06. If the cause is the edge, Δ(1040) stays near 0 and the positions far from the edges gain more.
- **Declared flaws.** The positions 1000 and 1500 ms come from another run with the same code and the same model. The 300 clips are those of R3, so the sweep is a second look at the same data and not a replication. The position 1040 ms leaves the 60 ms crop 10 ms from the edge.

### Result of the sweep (2 November)

Run [`omni-seam-v1`](../runs/omni-seam-v1/) on Kaggle with 2 T4, Qwen3-Omni at revision `26291f79` with NF4. These are 5,100 passes of 3.99 s each, finished at 00:29 UTC on 3 November. The reading is `tools/omni_seam.py` on that run and on `omni-control-v1`, on the 300 clips at 20, 40 and 60 ms. The AUROC at 20 ms with the centre at 500 ms (0.636) agrees with the one the run computed. The run's `verdict.json` applies the pilot rule, which requires 1000 ms, and does not correspond to this sweep.

| Centre of the crop | Where it falls | AUROC of `ab` | Δ against the seam | Encoder probe |
| --- | --- | --- | --- | --- |
| 500 ms | centre of the first block | 72.7 | +4.6 [2.9, 6.3] | 93.6 |
| 750 ms | first block, 250 ms from the seam | 71.4 | +3.3 [1.5, 5.0] | 93.5 |
| 1000 ms | the seam, split between two blocks | 68.1 | | 93.4 |
| 1040 ms | whole within the first token of the second block | 68.6 | +0.5 [−1.1, 2.2] | 93.3 |
| 1250 ms | second block, 250 ms from the seam | 72.8 | +4.7 [3.1, 6.2] | 93.4 |
| 1500 ms | centre of the second block | 74.7 | +6.6 [4.8, 8.6] | 93.5 |
| 1750 ms | second block, 250 ms from the end | 73.0 | +4.9 [3.0, 6.8] | 93.5 |

| Rule | Measure | Value | Reading |
| --- | --- | --- | --- |
| R5a | Lower limit of Δ at 500, 750, 1250, 1500 and 1750 ms | from 1.5 to 4.8 | the seam harms at all tested positions |
| R5b | AUROC at 500 ms minus at 1500 ms | −2.0 [−3.2, −1.0] | the position in the container also matters |
| R5c | Δ(1040) against half of Δ(1500), 3.3 | +0.5 [−1.1, 2.2] | splitting the crop does not explain most of it |
| R5c | AUROC at 1500 ms minus at 1040 ms | +6.1 [4.3, 8.0] | being at the edge of a block harms |

**The edge and not the split.** Evidence. With the crop whole within the first token of the second block the AUROC stays where it was on the seam. What harms is falling on the edge token of a block, split or whole. The prediction of the edge is met. Δ(1040) stays near 0 and within each block the centre gains more than the positions 250 ms from an edge, 6.6 against 4.7 and 4.9 in the second block and 4.6 against 3.3 in the first. The declared flaw of 1040 ms does not change the reading, because at 20 and 40 ms, with the crop 30 and 20 ms from the edge, the same happens (61.5 and 69.2 against 60.2 and 68.8 on the seam).

**The encoder loses no evidence.** Evidence. The regional encoder probe gives 93.3 to 93.6 at all positions, and no difference with the seam exceeds 0.3 points or excludes 0. The edge token keeps the evidence. A linear probe with labels reads it and the untrained readout of the language model does not. Inference on the mechanism. The first token of each block comes out of a convolution with zero padding on the left and with the position embedding reset, and that signature confuses the frozen readout. Which of the two weighs more is not settled.

**The position in the container.** Evidence for the effect, hypothesis for the cause. The first block ranks worse than the second, by 2.0 [1.0, 3.2] points at the centre and 1.6 [0.4, 2.8] at 250 ms from the end of each block. The sweep does not separate two causes. The tokens of the second block lie closer to the question in the language model's sequence, and the first block contains the start of the audio. The native BA goes the other way, 1.6 [0.4, 2.7] points higher at the centre of the first block, so the position also shifts the criterion.

**The gap with Qwen2-Audio.** Evidence. In the mean of 20, 40 and 60 ms the AUROC of the Qwen2-Audio base model exceeds that of Qwen3-Omni with `ab` by 11.4 [4.3, 18.4] points on the seam and by 4.8 [−1.6, 11.2] at the centre of the second block. With L1 the BA gap goes from 8.7 [3.6, 13.8] to 4.4 [−0.2, 9.4]. With 300 clips what remains cannot be distinguished from 0. The native rule rises little, from 50.6 to 54.4, because at 20 and 40 ms Qwen3-Omni answers "non-speech" to almost everything at any position.

**The study prompt hardly feels the seam.** Evidence. The registered score compares "S" with "N", but with that prompt the most probable first token is "NON" in 463 of the 600 items at 20 and 40 ms on the seam and in 545 at the centre of the second block, and "S" or "N" in the rest. To read the model's decision, `tools/omni_non.py` recomputes the logits of "S", "N" and "NON" from the stored final state and three rows of the output head of the same revision, downloaded by byte range. They reproduce `zA` and `zB` with a maximum error of 0.008. It is a descriptive decided after seeing the sweep.

| Study prompt, 20 and 40 ms | Seam | Centre of the second block | Difference |
| --- | --- | --- | --- |
| AUROC, "S" against "N" (registered) | 69.8 | 69.8 | +0.0 [−3.8, 3.6] |
| AUROC, "S" against "N" and "NON" | 69.8 | 71.0 | +1.2 [−2.5, 4.8] |
| Native BA, "S" against "N" and "NON" | 50.0 | 52.7 | +2.7 [1.5, 4.0] |
| AUROC of `ab`, for comparison | 64.5 | 71.1 | +6.6 [4.3, 9.1] |

With its prompt and the model's decision, Qwen3-Omni on the seam detects no speech at 20 and 40 ms (BA 50.0, the value the study reports at 20 ms), and at the centre of the second block it detects some (52.7). At that centre the two prompts rank equally, 71.0 against 71.1. The 5.3-point advantage of the study prompt on the seam comes from the edge harming `ab` and hardly harming that prompt. Untested hypothesis on why. With the letters A and B the language model would take the evidence from one token, and with the study's question it would integrate it over more tokens. Declared flaw. The study read generated text. With its prompt the most probable token of the first step is always "S", "N" or "NON", so the first-token decision is that of the generation unless a repetition penalty changes it, as in Qwen2-Audio. Which configuration the study generated with on Qwen3-Omni was not checked.

**For the study under review.** Its bank centres the crop in 2 s, on the seam. With its prompt the seam changes little for Qwen3-Omni at 20 and 40 ms, 2.7 points of native BA and a ranking without a clear difference. Its 0.50 at 20 ms is criterion at any position. With the question in A/B format the same position costs more than half of the ranking gap with Qwen2-Audio. A limitation note is enough. The Qwen3-Omni encoder processes 1 s blocks, the centred crop falls on the edge of two blocks, and with the study prompt that changes the result little at 20 and 40 ms. At 60 and 100 ms it is not measured with that prompt.

**What the paper says.** If it goes in, it goes in as an exploratory observation with a single backbone. In Qwen3-Omni part of the failure on short crops is due to design, the edge of the encoder's 1 s blocks, and the evidence is in the encoder at any position, as in Qwen2-Audio.

**Causes, alternatives and ideas.**

| Result | Cause and label | Alternative not ruled out | Idea, cost and test |
| --- | --- | --- | --- |
| Falling on the edge of a block harms the untrained readout | The edge token comes out of a convolution with zero padding and with the position reset. Evidence for the edge, inference for the mechanism | That only the padding or only the position weighs | A training-free fix for real use. Two passes with the audio shifted by 500 ms and the maximum of the two scores, so that every event lies far from an edge in one of them. The pairs 500 ms apart are already in the sweep, so it can be read on CPU in minutes, with a rule before |
| The first block ranks 2 points worse than the second | Proximity to the question or start of the audio. Hypothesis | Both | A 3 s container with the crop at the centre of each block. If the third block beats the second, it is proximity. About 2 h of Kaggle plus the generator, low priority |
| The study prompt hardly feels the seam | It integrates the evidence over more tokens. Hypothesis | That the effect exists and 300 clips are not enough (+1.2 [−2.5, 4.8]) | The study prompt at 1040 and 500 ms, about 40 min of Kaggle, with a rule before |
| Off the seam the gap with Qwen2-Audio cannot be distinguished from 0 | Wide interval with 300 clips. Evidence for the interval | 80 ms tokens, NF4 | None new. More clips only if Qwen3-Omni enters the paper with a claim about that gap |
