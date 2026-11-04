# Closing analyses, results snapshot

These files are the outputs of the exploratory analyses that `docs/exploratory_log.md` reports. The tools write them to `_work/closing/`, which is not shipped. Tables, JSON and CSV files that a tool writes to `_work/closing/interv/` are collected here in one folder. A `.md` file is the standard output of the command, saved as it was printed. "Post hoc" marks a check run with inline code that is not part of the repository. The tools also read features and per-item stores that are not shipped. The per-item scores of the full run come from `runs/`.

Commands are `PYTHONPATH=src python tools/<tool>`. `B = 2000` is the default number of bootstrap replicates.

| File | Written by | Section of `docs/exploratory_log.md` |
| --- | --- | --- |
| `tr_ci_base.csv`, `tr_ci_lora.csv` | `closing_analyses.py tr` | Step 8. L1_tr, a threshold learned on the training bank |
| `csweep_contrasts.csv` | `closing_gpu.py csweep` | Sensitivity of the matched-supervision head to regularisation |
| `figure_auroc_diff.csv` | `closing_extra.py all` | Numbers cited by the writing plan |
| `matched122.md` | `closing_extra.py matched` | Result of the matched-supervision reference |
| `heads_o.json`, `heads_f.json` | `intervention_analysis.py orig` (o, unfiltered audio) and `cue` (f, band-passed audio). Parameters of the cached heads. The `.npz` caches are not shipped | CPU answers to W2 and W3 of the review, Result of I1, intervention on channel cues |
| `orig.md` | `intervention_analysis.py orig` | CPU answers to W2 and W3 of the review |
| `w2.md` | `intervention_analysis.py w2` | CPU answers to W2 and W3 of the review |
| `negatives_report.md` | `screen_negatives.py` | Doubtful negatives of the provenance bank (W4 of the review) |
| `negatives/screen.csv` | `screen_negatives.py` | Doubtful negatives of the provenance bank (W4 of the review) |
| `negatives.md` | `intervention_analysis.py negatives` | Doubtful negatives of the provenance bank (W4 of the review) |
| `quiet.md` | `intervention_analysis.py quiet` | Exclusion of silent crops in both banks |
| `negatives/listening.csv` | `blind_listening.py --read` | Blind listening to the provenance negatives |
| `listened.md` | `intervention_analysis.py listened` | Blind listening to the provenance negatives |
| `clean_refit.md` | post hoc | Blind listening to the provenance negatives (row with heads refitted without the recordings heard as speech) |
| `cue.md` | `intervention_analysis.py cue` | Result of I1, intervention on channel cues |
| `split.md` | `intervention_analysis.py split` | Result of I2, LoRA restricted to part of the model |
| `csweep_contrasts_prov-v3-clean.csv` | `ERCLM_PROV_RUN=prov-v3-clean closing_gpu.py csweep` | Provenance with clean negatives |
| `figure_auroc_diff_prov-v3-clean.csv` | `ERCLM_PROV_RUN=prov-v3-clean closing_extra.py all` | Provenance with clean negatives |
| `cleanjoint.md` | `intervention_analysis.py cleanjoint` | CPU analyses requested by the review of draft v5 |
| `splitextra.md` | `intervention_analysis.py splitextra` | CPU analyses requested by the review of draft v5 |
| `figure122.md`, `figure_v5_supervision.csv` | `intervention_analysis.py figure`. Data of the figure of draft v5 (`make_figures.py --v5`) | CPU analyses requested by the review of draft v5 |
| `i3.md` | `intervention_analysis.py i3` | Result of I3 |
| `crossfilter.md` | `intervention_analysis.py crossfilter` | Follow-up of the review of draft v5 |
| `S.md` | `operating_point.py s` | Results of S, R and N |
| `R.md` | `operating_point.py r` | Results of S, R and N |
| `N.md` | `operating_point.py n` | Results of S, R and N |
| `S_extra.md`, `N_extra.md` | post hoc follow-ups of S and N | Results of S, R and N |
| `G1.md` | `operating_point.py g1` | Result of G1 |
| `G1_posthoc.md` | post hoc | Result of G1 |
| `G1_filtro.md` | post hoc, three checks of a reviewer's claims | Consolidated second reading of G1 (25 October) |
| `G1_transfer.md` | `operating_point.py g1t` | Consolidated second reading of G1 (25 October) |
| `P.md` | `operating_point.py p` | Result of P |
| `P_sweep.csv` | `operating_point.py p --sweep-csv`. Data of the figure of draft v6 (`make_figures.py --v6`) | Result of P |
| `P_sweep_only.md` | `operating_point.py p --B 0`, the sweep without the bootstrap | Result of P |
| `P_causas.md` | post hoc, causes of rule P | Result of P, Second reading of P and framing of draft v6 (25 October) |
| `methods.md` | `operating_point.py methods` | Second reading of P and framing of draft v6 (25 October) |
| `G2.md` | `operating_point.py g2` | Result of G2 |
| `G2_causas.md` | post hoc, causes of G2 | Result of G2 |
| `P_seg.md` | `operating_point.py p2` | Result of G2 |
| `perdur.md` | `operating_point.py perdur` | Response to the calibrated review |
| `perdur_causas.md` | post hoc, per-condition diagnostics | Response to the calibrated review |
| `V.md` | `operating_point.py v` | Result of V |
| `V_causas.md` | post hoc, causes of V | Result of V |
| `crossing.md`, `crossing.csv` | `operating_point.py crossing --sweep-csv`. Data of the crossing figure (`make_figures.py --v6`, `--v7`) | The fine-tuned model calibrated in logit space |
| `lora_cal.md` | `operating_point.py lora_cal` | The fine-tuned model calibrated in logit space |

Some file names keep the stems they had when they were written (`_causas` means causes, `_filtro` means filter). `negatives_report.md` was saved as `negativos_report.md`.

The `label` column of `negatives/listening.csv` holds the values the first author entered during the blind listening, in Spanish: `voz` (clear speech), `voz_posible` (possible or background speech), `vocal_no_verbal` (singing, laughter or another vocal sound) and `sin_voz` (no speech). The tools read these values as they are.
