# Running the experiments

This document describes how the experiments were executed and how a reader can run them. Everything goes through one program, the runner (`src/erclm/runner.py`), driven by a YAML file in `configs/`. Heavy artefacts (audio banks, features, adapters, checkpoints, full logs) go to a storage remote reached through rclone, or to a local folder. Small outputs (environment, progress, results, errors) form the record of each run, which this release ships in `runs/<run_id>/`. How the audio banks are built is in `docs/banks.md`.

## Requirements

- **Python 3.13**, the version of the CI and of every recorded run (3.13.15 or 3.13.16 on Colab and Kaggle).
- **Packages.** `requirements.txt` is for GPU machines, where the platform image already provides torch, numpy, pandas and scipy. It pins `transformers==5.19.0`, `peft==0.21.2` and `pyroomacoustics==0.10.1`, and asks for `accelerate>=1.10`, `bitsandbytes>=0.45`, `soundfile`, `librosa`, `scikit-learn` and `pyyaml`. `requirements-ci.txt` is for CPU machines and the tests. It pins the same three packages, leaves out bitsandbytes and librosa, adds `safetensors`, `tokenizers`, `numpy`, `pandas`, `scipy`, `pytest` and `nbformat`, and expects torch from the CPU index. Each run records the versions it used in `env.json` (for example torch 2.11.0, accelerate 1.14.0 or 1.15.0 and bitsandbytes 0.50.2).
- **rclone** for a remote storage (the launcher installs v1.75.2). Not needed with a local folder.
- **GPU.** The runs used a free Colab T4 (one GPU, reported as 14.56 GB) and Kaggle with two T4. The model is split over both GPUs on Kaggle (`device_map="auto"`). Bank building is CPU only.
- **Model.** `Qwen/Qwen2-Audio-7B-Instruct` at revision `0a095220c30b7b31434169c3086508ef3ea5bf0a`, the Hugging Face commit current on 15 October 2026 (last modified 12 January 2025). Every configuration pins it, and the runner refuses a configuration without a pinned revision. It is downloaded at every launch and never stored on the remote.
- **Precision.** 4-bit NF4 with double quantisation and FP16 compute. The modules that bitsandbytes leaves unquantised are also loaded in FP16, because the checkpoint is in bf16 and the T4 has no native bf16. Batch size one.

The adapter is loaded with `lib_v2.load_lora`, which remaps the encoder LoRA keys that an adapter saved with transformers 4.x loses silently under 5.x, and `lib_v2.lora_report` checks the result. Loading an adapter with `PeftModel.from_pretrained` directly skips that remapping.

The Qwen3-Omni runs use `Qwen/Qwen3-Omni-30B-A3B-Instruct` at revision `26291f793822fb6be9555850f06dfe95f2d7e695`, in NF4 on two T4. Their configurations pin `transformers==4.57.6` under the key `pip`, because transformers 5 fuses the experts and bitsandbytes does not quantise them. On Colab and Kaggle the runner installs those pins and restarts itself once.

FP16 results differ slightly between GPUs and CUDA versions. The model check on the synthetic container gave s = −3.789 on Colab (one T4, CUDA 13.0) and s = −3.781 on Kaggle (two T4, CUDA 12.8). Every extracted item records the launch and host that produced it. In the confirmatory run each model was extracted in a single launch on one platform, the base model on Colab and the fine-tuned model on Kaggle.

## Configuration

A configuration file overrides the defaults in `runner.DEFAULTS`.

| Key | Meaning |
| --- | --- |
| `run_name` | Names the run, its folders on the remote and its record. Required |
| `repo` | GitHub repository whose `runs` branch receives the run record |
| `remote` | rclone remote and folder for the heavy artefacts |
| `model` | `id`, `revision` (a pinned commit, required) and `quant` (`nf4`) |
| `prompt_key` | A prompt of `lib_v2.PROMPTS` or `lib_v3.PROMPTS`. `ab` is the confirmatory prompt |
| `llm_layers` | Language-model layers whose states are stored (8, 16 and 24) |
| `backbones` | `base`, `lora` or both |
| `adapter` | Remote folder of the LoRA adapter |
| `data` | For `test`, `train` and `dev`, the bank tar, the manifest and audio paths inside it, and the expected counts |
| `selection` | `pilot` (a subset of `n_per_class` excerpts per class, at most one per recording, drawn with `seed`) and `axes` (keep only some axes) |
| `extraction` | `shard_size` (items per shard), `n_content_free` (padding-noise containers added to the evaluation, plus one of digital silence), `n_head_checks`, `log_every` |
| `check` | `head_tol`, the tolerance of the logit reproduction check (0.01) |
| `analysis` | `bootstrap_B`, `seed`, `h3_margin`, `prevalence_reps` |
| `heartbeat` | `log_every_s` (300) and `progress_every_s` (1200) |
| `stages` | Stages to run when `--stages` is not given |
| `banks`, `provenance`, `retrain`, `omni` | Sections that enable `build_banks`, `build_prov`, `train_lora` and `extract_omni` |
| `audio_transform`, `lora_keep`, `generate` | Exploratory options. A transform of every container before the model, the LoRA kept only in part of the model, and a generation readout |
| `pip`, `env` | Extra packages for Colab and Kaggle, and environment variables set before torch is imported |

## The runner

```
PYTHONPATH=src python -m erclm.runner --config configs/<name>.yaml [options]
```

| Option | Effect |
| --- | --- |
| `--config` | The configuration file. Required |
| `--backbones base,lora` | Run only these backbones. The run id then gets a suffix, for example `full-v1-base`, so that two launches in parallel keep separate records |
| `--stages a,b` | Run only these stages, in their fixed order |
| `--force STAGE` | Redo a stage, deleting its stored outputs and markers. Repeatable, or `all` |
| `--remote-dir DIR` | Use a local folder as the remote instead of rclone (also `ERCLM_REMOTE_DIR`) |
| `--no-runs` | Do not publish the run record to GitHub |
| `--work DIR` | Local working folder (also `ERCLM_WORK`). The default is `/content/erclm_work` on Colab, `/tmp/erclm_work` on Kaggle and `./_work` elsewhere |
| `--plan` | Print the stages that would run and exit |

`ERCLM_HOST` forces the host detection (`colab`, `kaggle` or `local`) and `ERCLM_REPO` overrides the `repo` key.

### Stages

| Stage | What it does |
| --- | --- |
| `smoke` | Uploads, downloads and verifies an 8 MB file on the remote. Runs only when named, and is never marked done |
| `build_banks` | Builds the main, training and validation banks (`docs/banks.md`). CPU |
| `build_prov` | Builds the provenance bank. CPU |
| `train_lora` | Trains the LoRA on the training bank with the recipe of the study under review (`erclm.lora_train.RECIPE`) and keeps the checkpoint with the lowest loss on the validation bank, evaluated every 200 steps and at the end |
| `data_test`, `data_train` | Download the bank, check its sha256, validate counts, balance and WAV format, and record the result |
| `extract_omni` | Frozen readouts of Qwen3-Omni, one store per track, and the condition table of the registered rule |
| `check_base`, `check_lora` | Load the model (and the adapter) and run a fixed synthetic container, a 200 ms harmonic burst centred in 2 s of padding. They fail if logits or states are not finite, if 2 s do not give 50 audio tokens, if the logits computed from the captured final state differ from the model's by more than `head_tol`, or, for the fine-tuned model, if any LoRA B matrix is zero or the score equals the base model's |
| `extract_<b>_eval` | One forward pass per evaluation item, plus the content-free containers. Stores the logits of the two answer letters, the final state, the encoder output (mean over the clip and over the crop region, and maximum over the crop region), the projector output over the crop region and the configured language-model layers |
| `extract_<b>_train` | The same on the training bank, for the readout L2 and the encoder probe |
| `analyze_<b>` | Readouts, tables, bootstrap and the per-item score table `scores.csv.gz`. CPU |
| `joint` | Paired bootstrap of the two backbones from their score tables, for H3 and H4. CPU |

Without `--stages` or `stages` in the configuration, the runner plans every data, check, extraction and analysis stage that the configuration supports, for the selected backbones. GPU stages come before the CPU analyses, so an interruption during an analysis costs no GPU work. `joint` is planned only when the analysis of every other backbone is already done on the remote. The bank, training, Qwen3-Omni and smoke stages run only when named.

`--plan` shows the plan without running anything. With a local remote and `--no-runs` it needs no secret.

```
PYTHONPATH=src python -m erclm.runner --config configs/pilot.yaml --remote-dir _work/drive --no-runs --plan
```

For `configs/pilot.yaml` it prints `data_test`, `check_base`, `extract_base_eval`, `check_lora`, `extract_lora_eval`, `analyze_base`, `analyze_lora` and `joint`.

### A local run

On a machine with a CUDA GPU, a local folder can stand in for the remote. These commands build the banks, train the LoRA and run the pilot, each reading what the previous one wrote into `_work/drive`.

```
export PYTHONPATH=src
python -m erclm.runner --config configs/banks.yaml   --remote-dir _work/drive --no-runs
python -m erclm.runner --config configs/retrain.yaml --remote-dir _work/drive --no-runs
python -m erclm.runner --config configs/pilot.yaml   --remote-dir _work/drive --no-runs
```

`configs/full.yaml` runs the confirmatory analysis in the same way. `configs/smoke.yaml` checks the remote and the base model without any bank. `HF_TOKEN` in the environment is optional and avoids the slower downloads that Hugging Face applies without one.

The recorded costs give an idea of the scale. The full extraction of the main bank (21,340 variants plus 9 content-free containers) took 0.555 s per item on a Colab T4 for the base model and 0.667 s per item on Kaggle for the fine-tuned model, about 3.3 and 4.0 hours. Each analysis took about 21 minutes on CPU with 10,000 bootstrap replicates. The LoRA training took 20,519 s for 576 steps on two T4.

### Resuming

Every stage can be repeated and resumes where it stopped.

- A stage writes `state/<run_name>/<stage>.started.json` when it starts and `.done.json` when it ends. Both hold the hash of the part of the configuration that affects the stage, that part itself, the code version and the launch.
- A stage marked done with the same hash is skipped. An extraction marked done is first checked against its stored shards, and only the missing items are extracted again.
- A stage started or done with a different configuration is handled by kind. An extraction, of Qwen2-Audio or of Qwen3-Omni, stops the run and lists what changed, and only `--force <stage>` or a new `run_name` goes past it. Cheaper stages (data, checks, analyses) are redone with a warning.
- Extraction writes shards of `shard_size` items. Each shard is uploaded in the order npz, receipt, CSV, and the receipt holds the size and MD5 of the npz and the CSV. A shard counts as done only when its CSV is on the remote. On resume the runner lists the shards, checks every npz and CSV against its receipt without downloading it, removes leftover temporary files and orphans, and drops any shard that fails, CSV first, so its items are extracted again. An interruption costs at most one shard plus the model load.
- LoRA training uploads a checkpoint every 50 steps to `checkpoints/<run_name>/` and commits it with a `COMMIT.json` written last. A new launch resumes from the last committed checkpoint.
- If GitHub does not answer when a stage ends, the stage is marked done with `published: false` and the next launch republishes its outputs from the remote. At start, by contrast, the push of `env.json` must succeed, so that a bad GitHub token stops the run before any GPU work.

The log of each launch is written locally, redacting every secret (including the tokens inside the rclone configuration), and uploaded to the remote every 300 s. `progress.json` is published every 1200 s and at every stage boundary, with items done, seconds per item and estimated time left. If an exception escapes, the runner writes `error.md` with the traceback, the environment and the last 200 lines of the log, to the remote and to the run record.

## Storage

`src/erclm/storage.py` defines the remote. `RcloneRemote` works with any rclone remote. `LocalRemote` is a folder with the same behaviour, used by `--remote-dir` and by the tests. Every write goes to `<name>.part`, is renamed to `<name>` and is then checked for size and MD5 against the local file, with retries. A file counts as stored only after that check. Downloads are checked against the remote listing in the same way. Nothing heavy is in this repository.

| Remote path | Content |
| --- | --- |
| `data/` | Packed banks with their `.sha256`, bank metadata and protocol records under `manifests/`, simulated rooms under `rirs/` |
| `models/<name>/` | Adapters, for example `models/lora_retrained_v1/` |
| `checkpoints/<run_name>/` | Training checkpoints |
| `work/<run_name>/<backbone>/<eval or train>/` | Feature shards `shard_NNNNN.npz`, `.json` (receipt) and `.csv` |
| `work/<run_name>/<backbone>/results/`, `work/<run_name>/joint/` | Analysis outputs |
| `state/<run_name>/` | Stage markers |
| `logs/<run_id>/` | Full logs and error reports of every launch |

## Run records

Each run published a small record to an orphan `runs` branch of the repository, through `src/erclm/gitlog.py`. A publish fetches the tip of the branch, writes only inside `<run_id>/`, commits and pushes, retrying with growing waits, so two runs never conflict. The token travels in an HTTP header and never appears in a command line, a URL or a git configuration file. This release ships those records in `runs/<run_id>/`.

| File | Content |
| --- | --- |
| `env.json` | Last launch. Run id, launch id, host, start time, code version (`code_sha`, a commit of the development repository, whose history is not published), configuration file and hash, full configuration, model, stages, GPU and package versions |
| `sessions/<id>.json` | A copy of `env.json` for every launch. The id is `<UTC time>-<host>-<4 hex digits>` |
| `progress.json` | Status, timing and warnings of every stage |
| `<stage>.json` | Stage reports, for example `smoke.json`, `data_test.json`, `checks_base.json`, `extract_base_eval.json`, `checks_omni.json` |
| `results/<backbone>/`, `results/joint/`, `results/omni/` | Analysis tables, `summary.json` and the score tables `scores.csv.gz` |
| `banks/` | Bank records (`docs/banks.md`) |
| `retrain/report.json` | Validation losses, chosen checkpoint and LoRA report of the training |
| `error.md`, `errors/<id>.md` | The error of a failed launch. A new launch deletes `error.md` and keeps the copy in `errors/` |

| Run id | Configuration | What it is |
| --- | --- | --- |
| `smoke-v1` | `configs/smoke.yaml` | Checks of the remote (`smoke`) and of the base model on the synthetic container (`check_base`) |
| `banks-v2` | `configs/banks.yaml` | Builds the main, training and validation banks |
| `diag-banks` | `configs/diag_banks.yaml` | The bank stage again, with its remote in a separate folder, on a free Colab CPU machine, to find why long CPU jobs stopped there. It did not finish and no result uses it |
| `retrain-v1` | `configs/retrain.yaml` | Trains the LoRA. The validation loss was lowest at step 400 of 576 |
| `pilot-v1-base` | `configs/pilot.yaml` | Pilot, base model. 50 excerpts per class in the 22 conditions, extraction and analysis |
| `pilot-v1-lora` | `configs/pilot.yaml` | Pilot, fine-tuned model |
| `pilot-v1` | `configs/pilot.yaml` | Pilot, joint analysis |
| `full-v1-base` | `configs/full.yaml` | Confirmatory run, base model. The whole main bank and the training bank, extraction and analysis with 10,000 bootstrap replicates |
| `full-v1-lora` | `configs/full.yaml` | Confirmatory run, fine-tuned model |
| `full-v1` | `configs/full.yaml` | Confirmatory run, joint analysis (`docs/confirmatory_results.md`) |
| `prov-v3` | `configs/prov.yaml` | Builds the provenance bank (first launch, `--stages build_prov`) and extracts both models on it |
| `prompt-ab_speech_second` | `configs/prompt_ab_speech_second.yaml` | Prompt control. Speech labelled A and listed second, both models, on 150 excerpts per class in the 14 duration and SNR conditions (4,200 items) |
| `prompt-ba_speech_first` | `configs/prompt_ba_speech_first.yaml` | Prompt control. Speech labelled B and listed first |
| `prompt-ba_speech_second` | `configs/prompt_ba_speech_second.yaml` | Prompt control. Speech labelled B and listed second |
| `cue-main` | `configs/cue_main.yaml` | I1. The duration axis of the main bank with every container band-passed from 60 to 6000 Hz (Butterworth of order 8, zero phase) before the model, both models |
| `cue-prov` | `configs/cue_prov.yaml` | I1. The provenance bank through the same band-pass, both models |
| `split-llm` | `configs/split_llm.yaml` | I2. The LoRA kept only in the language model (128 modules), on the 4,200 items of the prompt control |
| `split-audio` | `configs/split_audio.yaml` | I2. The LoRA kept only in the audio encoder (96 modules) |
| `split-llm-lower` | `configs/split_llm_lower.yaml` | I2. The LoRA kept only in language-model layers 0 to 15 |
| `split-llm-upper` | `configs/split_llm_upper.yaml` | I2. The LoRA kept only in language-model layers 16 to 31 |
| `split-audio-llm-lower` | `configs/split_audio_llm_lower.yaml` | I3. The LoRA kept in the encoder and language-model layers 0 to 15 |
| `split-audio-llm-upper` | `configs/split_audio_llm_upper.yaml` | I3. The LoRA kept in the encoder and language-model layers 16 to 31 |
| `prompt-ba_speech_second-full` | `configs/prompt_ba_speech_second_full.yaml` | G1. Base model with speech labelled B and listed second, on the duration and SNR conditions of the whole main bank and on the training bank |
| `prompt-ab_speech_second-full` | `configs/prompt_ab_speech_second_full.yaml` | G2. The same with speech labelled A and listed second |
| `prompt-ba_speech_second-prov` | `configs/prompt_ba_speech_second_prov.yaml` | V. Base model with speech labelled B and listed second, on the provenance bank |
| `omni-pilot-v1` | `configs/omni_pilot.yaml` | First attempt at the Qwen3-Omni pilot. It stopped with a CUDA out-of-memory error while loading the quantised model (`error.md`) |
| `omni-pilot-v2` | `configs/omni_pilot.yaml` | Qwen3-Omni pilot, rerun with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The 100 pilot excerpts at 20, 40 and 1000 ms with three prompts |
| `omni-control-v1` | `configs/omni_control.yaml` | Qwen3-Omni on the 300 excerpts of the prompt control in 14 conditions, four tracks, 15,000 passes |
| `omni-seam-v1` | `configs/omni_seam.yaml` | Qwen3-Omni, sweep of the crop position across the encoder's two blocks of 1 s, at 20, 40 and 60 ms |
| `prompt-hand_csl` | `configs/prompt_hand_csl.yaml` | The hand-written prompt of the study under review, read at the first answer token, both models, on the 4,200 items of the prompt control |
| `prompt-opro_tmpl` | `configs/prompt_opro_tmpl.yaml` | The same with the prompt that OPRO-Template chose for the fine-tuned model in the study under review |
| `prompt-opro_llm_base` | `configs/prompt_opro_llm_base.yaml` | The same with the prompt that OPRO-LLM chose for the base model in the study under review |
| `gen-hand_csl` | `configs/gen_hand_csl.yaml` | Generation readout of the hand-written prompt on the same 4,200 items. Up to 16 new tokens, parsed as the study under review parsed them |
| `gen-opro_tmpl` | `configs/gen_opro_tmpl.yaml` | Generation readout of the OPRO-Template prompt |
| `gen-opro_llm_base` | `configs/gen_opro_llm_base.yaml` | Generation readout of the OPRO-LLM prompt |

The pilot only decided technical feasibility. All runs listed after `full-v1` are exploratory, and their rules and results are in `docs/exploratory_log.md` under the names used above (I1, I2, I3, G1, G2, V, the prompt control, the Qwen3-Omni pilot, control clips and sweep, the study prompts and their generation readout).

The confirmatory run was split across platforms. `--backbones base` ran on Colab (`full-v1-base`), `--backbones lora` ran on Kaggle (`full-v1-lora`), and `--stages joint` ran once both analyses were on the remote (`full-v1`).

### Recomputing from the score tables

Each analysis leaves a per-item score table with everything the bootstrap needs, so the joint analysis can be recomputed without a GPU or the remote.

```python
from erclm import analysis, export
tables = {b: export.load_scores(f"runs/full-v1-{b}/results/{b}/scores.csv.gz") for b in ("base", "lora")}
out = analysis.joint(tables, B=10000, seed=42, h3_margin=0.02)
out["tables"]["joint_contrasts"]
```

These are the settings of `configs/full.yaml`. With them the result reproduces `runs/full-v1/results/joint/joint_contrasts.csv` to floating-point precision (largest difference about 1e−16), in about 24 minutes on one CPU. A smaller `B` runs faster with wider Monte Carlo error.

## Colab and Kaggle

`notebooks/launcher.ipynb` is a three-cell launcher with no experiment logic. It is generated by `tools/build_launcher.py`, so a change goes into the builder and `python tools/build_launcher.py` rewrites the notebook. `tests/test_launcher.py` checks that the two match.

1. Cell 1 sets `CONFIG` (the name of a file in `configs/`), `CODE_REF` (default `main`), `EXTRA_ARGS` (runner options, for example `--backbones base` or `--force analyze_base`) and `REPO` (default `gbibbo/erclm-2027`). It detects the platform, checking Kaggle first because its image also sets Colab's variables, checks internet access, reads the secrets, copies them into the environment because the runner is a subprocess, and installs rclone. It prints only whether each secret is present, never a value.
2. Cell 2 fetches the repository at `CODE_REF`, records the commit and installs `requirements.txt`.
3. Cell 3 runs `python -m erclm.runner --config configs/<CONFIG>.yaml <EXTRA_ARGS>` and streams its output.

If the machine is lost, running all cells again resumes the run.

| Secret | Use |
| --- | --- |
| `HF_TOKEN` | Optional. Hugging Face token for faster model downloads |
| `RCLONE_CONF` | The content of an rclone configuration file with the remote named in the configuration's `remote` key, as pasted or encoded in base64 on one line (Kaggle's secret field may drop line breaks). Not needed with a local remote |
| `GH_TOKEN` | Optional. A GitHub token with write access to the contents of the repository that receives the run records. Without it, add `--no-runs` to `EXTRA_ARGS` |

To publish records from your own copy, point `repo` in the configuration (or `ERCLM_REPO`) to a fork with a `runs` branch.

On Colab the secrets go in the notebook's secret panel. On Kaggle they go in the notebook's secrets, with GPU and internet enabled, which requires a phone-verified account. Kaggle does not hand those secrets to runs started from its command line, so `tools/kaggle_push.py` reads them instead from a private dataset.

```
python tools/kaggle_push.py --make-credentials <file with a GitHub token> --user <Kaggle user>
python tools/kaggle_push.py --config smoke --ref main --user <Kaggle user> [--extra "--backbones base"] [--cpu] [--build-only]
kaggle kernels status <Kaggle user>/erclm-launcher
```

The first command creates or updates the private dataset `<Kaggle user>/erclm-credentials` from the local rclone configuration and the token, printing no secret. The second fills the launcher cells with the configuration, the code reference and the extra arguments, writes a private kernel with internet on Kaggle's T4 machine, which has two T4 (`--cpu` uses no GPU quota), and starts it with `kaggle kernels push`. It needs the Kaggle command-line tool 2.x.

Kaggle's GPU quota is weekly and Colab's free GPU is not always available. The banks were built on Kaggle because free Colab CPU machines stopped during that job. In the original work a job queue launched these notebooks on Colab and Kaggle. It is not part of this release.

## Tests

The tests run on CPU without downloads. `tests/test_model_v2.py` builds a tiny, randomly initialised Qwen2-Audio with the real processor and feature extractor classes and a toy tokenizer, so the chat template, audio-token expansion, state capture, feature store and LoRA loader run as on a GPU. The infrastructure tests use a local folder as the remote and simulate interruptions between uploads and during a shard, corrupt and duplicated shards, and stages done with another configuration. The tests of `RcloneRemote` are skipped when rclone is not on the `PATH`.

```
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-ci.txt
.venv/bin/python tests/test_lib_v2.py
.venv/bin/python tests/test_model_v2.py
.venv/bin/python -m pytest -q tests/test_storage.py tests/test_gitlog.py tests/test_runner.py \
  tests/test_launcher.py tests/test_bankgen.py tests/test_bankgen_v3.py tests/test_lora_train.py \
  tests/test_lib_v3.py tests/test_omni.py tests/test_omni_control.py tests/test_omni_seam.py \
  tests/test_genread.py tests/test_study_prompts.py tests/test_study_prompts_generation.py \
  tests/test_repetition_penalty.py
```

On Windows the interpreter is `.venv/Scripts/python`. The workflow `.github/workflows/tests.yml` installs the CPU dependencies and rclone on Python 3.13 and runs these suites on pushes and pull requests to `main`.
