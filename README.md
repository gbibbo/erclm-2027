# Does Fine-Tuning Make an Audio-Language Model Hear Speech Better?

Code, run records and the record of the study behind the paper of the same title by Gabriel Bibbó, Mark D. Plumbley and Arshdeep Singh, written for the ERC-LM workshop at ICASSP 2027.

The study asks whether fine-tuning Qwen2-Audio-7B-Instruct for speech detection makes it hear more, or only answer better. A detection passes through three steps: the model's internal states (the representation), the mapping from the final state to a preference between the answer letters (the readout), and the threshold on that preference (the criterion). The study changes one step at a time in the base model and applies the same change to the fine-tuned model, on clips whose speech is cut to 20 ms or buried in white noise down to −10 dB SNR. With the forced-choice prompt used to fine-tune it, the base model ranks speech above non-speech but answers at chance, mostly from a misplaced threshold that a label-free re-estimate largely corrects. Fine-tuning still improves ranking, and a linear readout trained on the fine-tuning data does not match it. Probes cross-fitted on the test conditions reach that ranking from the base model's final states.

The hypotheses, readouts, margin and statistics were fixed before any score was extracted. Everything else is exploratory and dated.

## Reading the paper with this repository

| Paper | Where to look |
| --- | --- |
| Sec. 3, banks (main, training, validation, provenance) | [docs/banks.md](docs/banks.md), `src/erclm/bankgen.py`, `src/erclm/bankgen_v3.py`, records in [runs/banks-v2](runs/banks-v2) and [runs/prov-v3](runs/prov-v3) |
| Sec. 3, blind listening that kept 76 of 114 provenance recordings | [docs/exploratory_log.md](docs/exploratory_log.md), "Blind listening to the provenance negatives", labels in [results/closing/negatives/listening.csv](results/closing/negatives/listening.csv) |
| Sec. 3, model and fine-tuning recipe | `src/erclm/lora_train.py`, `configs/retrain.yaml`, record in [runs/retrain-v1](runs/retrain-v1) |
| Sec. 3, prompt and prompt control | `PROMPTS["ab"]` in `src/erclm/lib_v2.py`, the control variants in `src/erclm/lib_v3.py`, `configs/prompt_*.yaml`, records `runs/prompt-*` |
| Sec. 3, measures, readouts L0 to L2, H1 to H4, margin of H3, statistics | [docs/study_design.md](docs/study_design.md), `src/erclm/lib_v2.py` (`cluster_bootstrap`, `combine_runs`, `fit_head`, `crossfit_scores`) |
| Sec. 3, "the record of the study holds the secondary hypotheses H2b and H4 and the dated history of every exploratory analysis" | [docs/confirmatory_results.md](docs/confirmatory_results.md) and [docs/exploratory_log.md](docs/exploratory_log.md) |
| Sec. 4.1, AUROC 87.5 and 92.0 against BA 50.1, best thresholds 81.5 and 86.4, L1 at 79.1 and 81.1, H1 | [docs/confirmatory_results.md](docs/confirmatory_results.md), condition tables in [runs/full-v1-base](runs/full-v1-base) |
| Sec. 4.1, Figure 1 | [figures/fig_v14_criterion.pdf](figures/fig_v14_criterion.pdf), written by `tools/make_figures.py --v14` from the condition tables of `runs/full-v1-base` and `runs/full-v1-lora` |
| Sec. 4.1, preference for the letter B | [docs/exploratory_log.md](docs/exploratory_log.md), "Prompt control" |
| Sec. 4.1, averaging the original and the swapped prompt (55.8 and 65.2 against 84.5 and 91.6) | "Results of S, R and N" in the log, [results/closing/S.md](results/closing/S.md), `tools/operating_point.py s` |
| Sec. 4.1, the fine-tuned model at 20 ms (24.8% of speech detected, 61.9 against 71.4 BA, 64.0 with L1) | condition table in [runs/full-v1-lora](runs/full-v1-lora); the saturation of L1 in "The fine-tuned model calibrated in logit space" |
| Table 1, rows L0, L1 and L2 | [docs/confirmatory_results.md](docs/confirmatory_results.md), "Means per axis" |
| Table 1, probe rows, and Sec. 4.3 (95.8 and 99.8, −0.06, 3.1, the fall of 8.4 points) | "Numbers cited by the writing plan" and "Step 12" in the log, `tools/layer_accessibility.py`, `tools/closing_extra.py`, `tools/closing_gpu.py` |
| Sec. 4.2, H3 and its dependence on the penalty (+1.1 to −9.4, −2.18) | [docs/confirmatory_results.md](docs/confirmatory_results.md); "Sensitivity of the matched-supervision head to regularisation" in the log, [results/closing/csweep_contrasts.csv](results/closing/csweep_contrasts.csv) |
| Sec. 4.3, H2a | [docs/confirmatory_results.md](docs/confirmatory_results.md) |
| Sec. 4.3, 122 excerpts per fold and the provenance bank (1.0, 0.4, 2.4 at 20 and 40 ms) | "CPU analyses requested by the review of draft v5" in the log, item N3 |

## Repository layout

| Path | Content |
| --- | --- |
| `src/erclm/lib_v2.py` | The analysis library of the pre-registered design: prompts, scores, readouts, heads, probes, cluster bootstrap, joint analysis. Frozen with the design |
| `src/erclm/lib_v3.py` | Exploratory analyses added after the design was frozen (prompt variants, coverage and level checks, the band-pass and LoRA-restriction interventions), with their own names. The confirmatory readouts are untouched |
| `src/erclm/runner.py` | Resumable runner: each config is a list of stages (bank building, feature extraction, LoRA training, analysis) |
| `src/erclm/bankgen.py`, `bankgen_v3.py`, `banks.py` | Bank generation from VoxConverse and ESC-50, packing and verification |
| `src/erclm/lora_train.py` | LoRA retraining with the recipe of the study under review |
| `src/erclm/analysis.py`, `export.py` | Per-backbone and joint analyses, per-item score tables |
| `src/erclm/genread.py`, `omni.py` | Exploratory readouts: generation readout of Qwen2-Audio, and Qwen3-Omni as a second model |
| `src/erclm/host.py`, `storage.py`, `gitlog.py` | Host detection, remote storage with verified uploads, publication of run records |
| `configs/` | One YAML per run |
| `tools/` | Exploratory analyses, figures, bank packing, the Colab and Kaggle launcher |
| `notebooks/launcher.ipynb` | Three-cell launcher for Colab and Kaggle |
| `runs/` | The record of every run ([runs/README.md](runs/README.md)) |
| `figures/` | Figure 1 of the paper, regenerated from `runs/` |
| `results/closing/` | Reports of the exploratory analyses ([results/closing/README.md](results/closing/README.md)) |
| `docs/` | [study_design.md](docs/study_design.md), [confirmatory_results.md](docs/confirmatory_results.md), [exploratory_log.md](docs/exploratory_log.md), [banks.md](docs/banks.md), [running.md](docs/running.md) |
| `tests/` | CPU tests with a tiny randomly initialised Qwen2-Audio |

## Reproducing

Install the CPU requirements and run the tests:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ci.txt
PYTHONPATH=src python -m pytest -q tests
```

The tests of the rclone remote are skipped when rclone is not installed. [docs/running.md](docs/running.md) lists the suites one by one, as the CI runs them.

The confirmatory contrasts can be recomputed without a GPU from the per-item scores in `runs/` ([docs/confirmatory_results.md](docs/confirmatory_results.md), "Recomputing these numbers"). Rebuilding the banks, retraining the LoRA and extracting the scores needs VoxConverse and ESC-50, which are not redistributed here, and a GPU with about 16 GB ([docs/banks.md](docs/banks.md), [docs/running.md](docs/running.md)). Audio, features, adapters and checkpoints are not in the repository.

## Citation

Please cite the paper: G. Bibbó, M. D. Plumbley and A. Singh, "Does Fine-Tuning Make an Audio-Language Model Hear Speech Better?", ERC-LM workshop, ICASSP 2027. The full reference will be added on publication.
