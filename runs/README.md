# Run records

Each folder is the record of one run, published by the runner while the run was executing on Colab or Kaggle. The records are kept as they were written. Heavy artefacts (audio banks, extracted features, adapters, checkpoints) are not here; only small outputs are. The table of runs, with their configs and purpose, is in [docs/running.md](../docs/running.md).

Typical contents of `runs/<run_id>/`:

| File | Content |
| --- | --- |
| `env.json` | Host, GPU, library versions, the resolved config, the model revision, the stages and `code_sha` |
| `progress.json` | Stage-by-stage progress, rewritten during the run |
| `sessions/<session>.json` | One record per session on a host; a run resumed after a cut has several |
| `<stage>.json` | Report of a stage (for example `extract_base_eval.json`, `checks_base.json`, `data_test.json`) |
| `results/<backbone>/` | Analysis outputs: `condition_table.csv`, `contrasts.csv`, `bootstrap_ci.csv`, `prevalence.csv`, `thresholds.csv`, `efficiency.csv`, `summary.json` and the per-item score table `scores.csv.gz` with `scores.json` |
| `results/joint/` | The joint analysis of both backbones (`joint_contrasts.csv`, `joint_bootstrap_ci.csv`, `joint_summary.json`) |
| `error.md`, `errors/` | Present only if a session failed: the stage, the time and the traceback |

`code_sha` identifies the commit of the development repository that produced the run. That repository is private. The code in this repository is its state at the time of release, with comments and documentation translated. The only functional changes are the repository name and the branch that the launcher clones. The resolved config in `env.json` shows exactly what each run executed.

The confirmatory results are in `full-v1-base`, `full-v1-lora` and `full-v1` ([docs/confirmatory_results.md](../docs/confirmatory_results.md)).
