# Audio banks

This document describes how every audio bank of the study was built, so that a reader can rebuild or audit them. The code is `src/erclm/bankgen.py` (main, training and validation banks, protocol `banks-v2.0`), `src/erclm/bankgen_v3.py` (provenance bank, protocol `prov-v3.1`) and `src/erclm/banks.py` (packing, download and validation). The configurations are `configs/banks.yaml` and `configs/prov.yaml`.

The audio is not in this repository. The records of the two runs that built the banks are in `runs/banks-v2/banks/` and `runs/prov-v3/banks/`. They hold the list of excerpts with their source file and start, the parameters of every variant, the simulated rooms and the validation report with the sha256 of each packed bank.

## Why the banks were regenerated

The study under review (Bibbó et al., under review at Computer Speech & Language) trained and tested its fine-tuned model on a test bank of 970 excerpts in 22 conditions and a training bank of 64 excerpts in 48 conditions. To locate the gain that it reported, this study needed the same setting. In October 2026 those banks were stored on a compute cluster that none of the authors could access. On 16 October 2026 the authors decided to rebuild them from the public corpora with the same design and the same sizes, and to retrain the LoRA on the new training bank with the recipe of the study under review (`configs/retrain.yaml`). The regenerated banks keep the design and the sizes, not the identity of the excerpts.

An audit of the generation code of the study under review, made on 15 October 2026, had shown two things. The banks could not be rebuilt bit for bit, because the padding and SNR noise had no seed, the starts of the excerpts were not stored, the speech windows were chosen with Silero VAD loaded from torch.hub without a pinned version, and the files passed through an unsorted glob and a Python set before shuffling. And three of their properties had to be fixed.

- **Reverberation.** The generator looked for a room impulse response (RIR) within 0.2 s of the target T60 and used the first one it found. Its RIR categories had typical T60 values of 0.3, 0.6 and 1.0 s, labels by room size rather than measured values. For 2.5 s there was none, so that condition was left dry. This was verified on the per-item predictions of the fine-tuned model of the study under review. The mean |Δp| against `reverb_none` of the same excerpt was 0.012 at 2.5 s, as for `dur_1000ms` (0.011) and `filter_none` (0.011), which are dry copies with other padding noise, against 0.038 at 0.3 s and 0.046 at 1.0 s. The 0.3 s and 1.0 s levels were one RIR each.
- **SNR.** The reference RMS for the noise included the padding of the 2 s container. Read from the code and not checked against audio, the effective SNR over a 1000 ms excerpt was about 3 dB above the nominal level, and in the training bank the gap grew as the excerpt shortened, to about 10·log10(2000/d) dB for an excerpt of d ms.
- **Shared recordings.** Verified. 60 of the 64 recordings of the training bank (30 from VoxConverse and 30 from ESC-50) were also in the test bank, and 123 of the 970 test excerpts (86 speech and 37 non-speech) came from them.

The audit also found, from the code, that training excerpts were peak-normalised to 0.9 with a 3 dB margin, about 0.64, while test excerpts were normalised to 0.9. The regeneration fixes all four. The full list of differences is in [Differences from the banks of the study under review](#differences-from-the-banks-of-the-study-under-review).

## Sources

| Corpus | Version | Content |
| --- | --- | --- |
| VoxConverse | v0.3. Audio of the dev and test sets from the URLs in `configs/banks.yaml`. RTTM annotations from `github.com/joonson/voxconverse` at commit `24bf60be297701cd7e4ef18550c6d390c1b87365` | 448 recordings with both audio and annotations |
| ESC-50 | `github.com/karolpiczak/ESC-50` at commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6` | 2,000 clips of 5 s in 50 classes, from 1,528 distinct sources |

The RTTM annotations and ESC-50 are pinned by commit. The VoxConverse audio archives are downloaded from their URLs and the code does not check them against a hash.

## Recordings and splits

The unit of separation is the source recording. In VoxConverse it is an audio file. In ESC-50 it is the pair of fold and freesound source file, which groups the takes A, B and C of one source. This is the `rec_id` that `erclm.lib_v2` derives from a file name (`vox:<id>` or `esc:<fold>-<source>`), and the unit of the cluster bootstrap.

`bankgen.select_clips` draws everything from one random generator seeded with 42.

- VoxConverse recordings are sorted, permuted and assigned in that order, 210 to the main bank, 32 to the training bank and 36 to the validation bank. A recording without enough usable windows is skipped.
- ESC-50 sources are sorted, permuted and assigned in that order, 485 to the main bank, 32 to the training bank and 36 to the validation bank. Each source gives one excerpt, from a take drawn at random. A source whose take has no usable window is skipped.
- The generator stops with an error if any two banks share a recording.

## Excerpts of 1000 ms

- **Speech.** Candidate windows of 1000 ms on a 100 ms grid, with at least 90% speech according to the RTTM (on 10 ms frames) and an RMS of at least 1e−3. Windows are drawn at random without overlap, with starts at least 2 s apart, so that two windows of one recording are separated by at least 1 s. In the main bank 65 recordings give three excerpts and 145 give two, 485 in all. In the training and validation banks each recording gives one.
- **Non-speech.** Candidate windows of 1000 ms on a 100 ms grid whose RMS is at least a quarter of the RMS of the loudest window of the clip, so that silence is not taken. One is drawn at random.
- **Processing.** Mixdown to mono, resampling to 16 kHz with a fixed polyphase filter (`scipy.signal.resample_poly`) when the source rate differs, removal of the DC offset and peak normalisation to 0.9. The same in every bank.
- The source file and the start of every excerpt are stored.

The paper reports that 99.8% of the speech crops from 20 to 100 ms fall entirely within annotated speech (`docs/exploratory_log.md`, "V1. Speech occupancy of the short crops").

## Container and conditions

Every variant is a 2000 ms container at 16 kHz (32,000 samples), written as a float32 WAV, with the excerpt or its crop centred in Gaussian padding of standard deviation 1e−4. `bankgen.render` builds a variant in this order.

1. Centred crop of the 1000 ms excerpt to the duration of the condition (1000 ms when the condition sets none), with raised-cosine ramps of 1 ms at both ends. The crop is not renormalised, so it keeps its level within the excerpt.
2. Reverberation, when the condition asks for it.
3. Filter, when the condition asks for it.
4. RMS of the resulting segment, stored as `segment_rms`.
5. Placement in the centre of the padded container.
6. Noise, when the condition sets an SNR. White Gaussian noise over the whole container with standard deviation `segment_rms / 10^(SNR/20)`, so that the level describes the excerpt and not the container.
7. Clipping guard. If the peak of the container exceeds 0.99, the whole container is scaled down, which leaves the SNR unchanged, and the gain is stored. In the main bank the guard scaled 3,768 of the 21,340 variants, 3,150 of them in the SNR conditions, with a smallest gain of 0.147.

| Axis | Levels | Operation |
| --- | --- | --- |
| Duration | 20, 40, 60, 80, 100, 200, 500 and 1000 ms | Centred crop in the near-silent container |
| SNR | −10, −5, 0, 5, 10 and 20 dB | Full excerpt plus white noise over the container |
| Reverberation | none, 0.3, 1.0 and 2.5 s | Convolution with the RIR of a room drawn at random among the eight of its level, cut to the length of the excerpt and scaled to the RMS of the dry excerpt |
| Filter | none, band-pass from 300 to 3400 Hz, low-pass at 3400 Hz, high-pass at 300 Hz | Butterworth of order 4 applied forward and backward (`sosfiltfilt`, zero phase) |

The main bank crosses each excerpt with the 22 one-factor conditions. The paper uses only the duration and SNR axes. The reverberation and filter axes are built and validated but not analysed. The conditions `dur1000ms`, `reverbnone` and `filternone` hold the same excerpt with independent padding noise, so their files differ (`data_test.json` of every run on the main bank reports 0 of 970 excerpts with identical files in the three).

The training and validation banks cross each excerpt with the 48 combinations of the 8 durations and the 6 SNR levels, so every container carries noise.

All randomness of a variant comes from its own seed, the integer value of the first 12 hexadecimal digits of the sha256 of `banks-v2.0|<base_id>|<condition>`, stored in the `seed` column. It drives the padding, the choice of room and the noise.

## Rooms for reverberation

`bankgen.make_rooms` simulates eight shoebox rooms per T60 level with the image-source method of `pyroomacoustics` 0.10.1, which is deterministic, from a generator seeded with `[42, 7]`.

| T60 target | Room dimensions (m) | Measured T60 of the eight rooms |
| --- | --- | --- |
| 0.3 s | 3 to 6 by 3 to 6 by 2.5 to 3.2 | from 0.276 to 0.324 s |
| 1.0 s | 6 to 12 by 5 to 10 by 3 to 4 | from 0.960 to 1.062 s |
| 2.5 s | 10 to 20 by 8 to 16 by 4 to 8 | from 2.253 to 2.687 s |

- Source and microphone are placed at random at least 1 m from every wall. Reflections are limited to order 60, without air absorption.
- The absorption starts from the inverse Sabine formula and is corrected, up to six times, by the ratio of measured to target T60. A room is kept only when its T60, measured by Schroeder integration (`measure_rt60`), is within 10% of the target. Otherwise another room is drawn.
- Each RIR is cut to start at its direct path (the first sample at 10% of the maximum) and normalised to unit energy.
- The rooms are stored once on the remote as `data/rirs/rooms_v2.npz` with a `.json` of their parameters, and every later build reads them from there. The parameters are in `runs/banks-v2/banks/rooms.json`.

## The banks

| Bank | Name | Excerpts | Recordings | Conditions | Variants | Used for |
| --- | --- | --- | --- | --- | --- | --- |
| Main | `test_bank_v2` | 970, 485 speech from VoxConverse and 485 non-speech from ESC-50 | 695, 210 from VoxConverse and 485 ESC-50 sources | 22 one-factor | 21,340 | All tests |
| Training | `train_bank_v2` | 64, 32 per class | 64 | 48 duration by SNR | 3,072 | Training of the LoRA and of the readouts fitted on it, among them L2 and the encoder probe of H2a |
| Validation | `dev_bank_v2` | 72, 36 per class | 72 | 48 duration by SNR | 3,456 | Choice of the LoRA checkpoint |
| Provenance | `prov_bank_v3` | 228, one speech and one non-speech window per recording | 114 from VoxConverse | 8 durations | 1,824 | Provenance control. The final analyses use the 76 recordings kept after the blind listening |

Every bank is balanced, with half of its variants speech. No two of the main, training and validation banks share a recording, and the provenance bank uses none of their VoxConverse recordings.

The run `banks-v2` built the first three on 19 October 2026 on a Kaggle CPU machine, in 1,536 s for the stage. The run `prov-v3` built the provenance bank on 21 October 2026 (its first launch, with `--stages build_prov`).

## Names and metadata

File names follow the scheme of the study under review, so `erclm.lib_v2` parses them unchanged into condition, axis, `base_id` and `rec_id`.

- A variant is `<base_id>_1000ms_<condition>.wav`, for example `voxconverse_abjxc_0012_1000ms_snr+0dB.wav` or `esc50_1-100032-A-0_0007_1000ms_reverb2.5s.wav`.
- `base_id` is `voxconverse_<recording>_<index>` or `esc50_<ESC-50 file stem>_<index>`, with a four-digit index that counts the excerpts of the bank in selection order.
- The conditions of the main bank are `dur20ms` to `dur1000ms`, `snr-10dB` to `snr+20dB`, `reverbnone`, `reverb0.3s`, `reverb1.0s`, `reverb2.5s`, `filternone`, `filterbandpass`, `filterlowpass` and `filterhighpass`. Those of the training and validation banks are `dur<d>ms_snr<s>dB`.

A bank folder holds three things.

- `manifest.csv`, with `audio_path` (file name only) and `ground_truth` (`SPEECH` or `NONSPEECH`). This is what the runner reads.
- `metadata.csv`, one row per variant, with `audio_path`, `ground_truth`, `base_id`, `split`, `dataset`, `recording`, `source_file`, `start_s`, `condition`, `duration_ms`, `snr_db`, `t60_target_s`, `t60_measured_s`, `room`, `filter`, `seed`, `segment_rms`, `gain` and `noise_sigma`. The provenance bank has no `noise_sigma`.
- `audio/`, with every WAV, float32, 2.000 s at 16 kHz.

The run records hold the list of excerpts (`clips.csv` and `prov_bank_v3.clips.csv`, with split, label, dataset, recording, source file, start and, for ESC-50, the class) and the metadata of each bank (`<name>.metadata.csv.gz`).

## The provenance bank and the blind listening

In the main bank every speech excerpt comes from VoxConverse and every non-speech excerpt from ESC-50, so corpus and class go together and a probe could separate the classes by corpus or channel rather than by speech. The provenance bank takes both classes from the same recordings. `bankgen_v3` only selects the excerpts and reuses `bankgen` for loading, rendering, naming and packing.

- **Recordings.** The VoxConverse recordings that no other bank uses, 170 of the 448. The exclusion list is the list of excerpts that the bank stage wrote (`provenance.exclude_clips` in `configs/prov.yaml`, the same content as `runs/banks-v2/banks/clips.csv`). The remaining recordings are sorted and permuted with seed 42, and a recording enters only if it has a usable window of each class. 114 did. The stage requires at least 100.
- **Speech window.** The rule of the other banks, 1000 ms with at least 90% annotated speech.
- **Non-speech window.** 1000 ms on the 100 ms grid with no annotated speech within 250 ms on either side, and an RMS of at least a quarter of the median RMS of the recording's speech windows (and at least 1e−3), so that it is not silence.
- **Conditions.** The 8 durations of the main bank, without noise, in the same container and padding. 228 excerpts and 1,824 variants.
- Both windows of a recording are named `voxconverse_<recording>_<index>`, so they share a `rec_id` and every resampling keeps them together.

The non-speech windows come from stretches without annotation, which may still hold speech. Two checks followed on 22 October 2026.

1. **Screen.** An AudioSet tagger (AST, `MIT/ast-finetuned-audioset-10-10-0.4593`) scored the 1000 ms variant of all 228 excerpts with `tools/screen_negatives.py`. It ranks the negatives by speech and singing probability and does not label them (`results/closing/negatives/screen.csv`).
2. **Blind listening.** The first author, blind to all scores, listened to the 1000 ms variant of the 114 non-speech windows on a local page built by `tools/blind_listening.py`, with neutral names in a fixed random order and no tagger score. The four categories were fixed before listening.

| Category | Windows |
| --- | --- |
| Clear speech | 25 |
| Possible or background speech | 13 |
| Singing, laughter or another vocal sound | 29 |
| No speech | 47 |

The recordings of the 38 windows with clear or possible speech are removed, each with its speech window, and the other 76 recordings (152 excerpts) are kept. Non-verbal vocal sounds count as non-speech, as the vocal classes of ESC-50 do in the main bank. The tagger missed 22 of the 38. The labels, joined with the tagger probabilities, are in the column `label` of `results/closing/negatives/listening.csv`. The bank itself is unchanged. `tools/clean_provenance.py` applies the exclusion to the features extracted from it. The effect on the results is in `docs/exploratory_log.md`, "Blind listening to the provenance negatives".

## Differences from the banks of the study under review

| Aspect | Study under review | This study |
| --- | --- | --- |
| Reverberation of 2.5 s | Dry, for lack of an RIR of that level | Applied, with measured T60 |
| RIRs per level | One | Eight |
| T60 | Label by room size | Measured |
| SNR reference | RMS of the 2 s container | RMS of the excerpt |
| Training and test | 60 of 64 training recordings also in the test bank | No shared recording |
| Peak normalisation | 0.9 in test and about 0.64 in training | 0.9 everywhere |
| Speech selection | Silero VAD without a pinned version, at least 80% speech | RTTM annotations, at least 90% speech |
| Test recordings | 650 (210 VoxConverse, 440 ESC-50 sources) | 695 (210 VoxConverse, 485 ESC-50 sources, one excerpt each) |
| Starts of the excerpts | Not stored | Stored |
| Noise and crops | No seed | One seed per variant |
| Duration crop | No ramps | Ramps of 1 ms |
| Clipping | Samples clipped | Container scaled down |

Both use 22 one-factor conditions in a 2 s container with Gaussian padding of standard deviation 1e−4, the same filters (Butterworth of order 4, zero phase) and the same reverberation procedure (convolution cut to the excerpt and matched in RMS).

## Rebuilding a bank

The bank stages run on CPU and need the packages of `requirements-ci.txt` (with `pyroomacoustics==0.10.1`). The sources take about 7 GB to download and the packed main bank about 2.78 GB. On free Colab CPU machines the bank stage stopped repeatedly after about 12 minutes and the cause was not isolated (`runs/diag-banks`), so build the banks on Kaggle or on a local machine.

This command builds the main, training and validation banks into a local folder that stands in for the remote.

```
PYTHONPATH=src python -m erclm.runner --config configs/banks.yaml --remote-dir _work/drive --no-runs
```

The stage downloads and unpacks the sources into `<work>/sources`, loads the rooms from the remote or simulates and stores them, selects the excerpts, renders, validates and packs each bank, and writes into the remote folder.

| Remote path | Content |
| --- | --- |
| `data/<name>.tar` and `data/<name>.tar.sha256` | The packed bank and its sha256 |
| `data/manifests/<name>.metadata.csv.gz` | Its metadata |
| `data/manifests/<name>.protocol.json` | Specification hash, specification and validation report |
| `data/manifests/clips_<hash>.csv` | The list of excerpts of the three banks |
| `data/rirs/rooms_v2.npz` and `.json` | The rooms |

A bank whose tar, sha256 and protocol record are already on the remote with the same specification is not rebuilt, so an interrupted build loses at most the bank in progress. The specification holds the protocol version, the seed, the names, the counts, the sources, the expected counts and a hash of the source of `bankgen.py`.

This one builds the provenance bank.

```
PYTHONPATH=src python -m erclm.runner --config configs/prov.yaml --remote-dir _work/drive --no-runs --stages build_prov
```

It reads the exclusion list from the remote path in `provenance.exclude_clips`. The `<hash>` in `clips_<hash>.csv` depends on the specification of the bank stage, which includes the hash of `bankgen.py`, so a rebuild with edited code writes another file name. Point `exclude_clips` to the file that the bank stage wrote, or copy `runs/banks-v2/banks/clips.csv` to the configured path.

The same steps can be run without the runner.

```python
import yaml
from erclm import bankgen as G, bankgen_v3 as G3

pins = yaml.safe_load(open("configs/banks.yaml"))["banks"]["sources"]
vox, esc, esc_paths = G.fetch_sources("_work/sources", pins)
rooms = G.make_rooms(seed=42)
clips = G.select_clips(vox, esc, seed=42)
G.build_bank("test_bank_v2", clips[clips.split == "test"], G.test_conditions(), vox, esc_paths, rooms, "_work/banks")
G.build_bank("train_bank_v2", clips[clips.split == "train"], G.factorial_conditions(), vox, esc_paths, rooms, "_work/banks")
G.build_bank("dev_bank_v2", clips[clips.split == "dev"], G.factorial_conditions(), vox, esc_paths, rooms, "_work/banks")
tar, sha = G.pack("test_bank_v2", "_work/banks")

used = set(clips.loc[clips.dataset == "voxconverse", "recording"])
prov = G3.select_provenance_clips(vox, used, seed=42, per_recording=1)
G.build_bank("prov_bank_v3", prov, G3.duration_conditions(), vox, {}, None, "_work/banks")
```

`tests/test_bankgen.py` and `tests/test_bankgen_v3.py` exercise these functions on synthetic audio.

**Comparing a rebuild with the original.** Selection and rendering are seeded and every input list is sorted, so a rebuild with the pinned sources should select the same excerpts. Compare the `ids_hash` of the validation report (a sha256 of the sorted variant names) and the columns `start_s`, `seed`, `segment_rms` and `gain` of the metadata with the records below. The sha256 of the tar is not a test of the content, because tar headers store file modification times, so a rebuilt tar has another sha256 even when every WAV is equal. Whether every WAV is bitwise equal also depends on the versions of the numerical libraries, which `runs/banks-v2/env.json` records for the original build. Both points are read from the code and were not tested.

## Packing and verification

- `bankgen.pack` writes `<name>.tar` with the members in sorted order, first under a temporary name and then renamed, and a `<name>.tar.sha256` with the line `<sha256>  <name>.tar`.
- Before packing, `banks.validate` checks the number of rows, conditions, `base_id` and `rec_id` against the expected counts in the configuration, the share of speech, that every file name has a recognisable condition, and that every WAV is at 16 kHz and lasts 2.000 s within half a sample. If a check fails, the stage publishes the report and stops before uploading that bank.
- Every upload goes to a temporary name, is renamed and is then checked for size and MD5 against the local file (`src/erclm/storage.py`).
- At use, `banks.fetch` downloads the tar, compares its sha256 with the `.sha256` on the remote, refuses links and paths outside the target folder, extracts it and leaves a marker named after the sha256, so the extraction happens once per bank version. The `data_test` stage validates the bank again and records its sha256. Every extraction stage includes that sha256 in its specification, so a changed bank stops a resumed extraction instead of mixing two banks.
- `banks.make_tar` packs any manifest and audio folder in the layout that `banks.fetch` reads, for a bank built elsewhere (`tools/pack_bank.py`).

## Records of the banks used

| Bank | Validation report | sha256 of the tar | `ids_hash` |
| --- | --- | --- | --- |
| `test_bank_v2` | `runs/banks-v2/banks/test_bank_v2.json` | `a12a6b28e065b15956537aa2e2bf162a47262a8e18497f0bbab83496c0ae6ea1` | `f4c0f780827d02ad4dac7d2525ebcfccfcaa67d7df6c238587118d10cc774128` |
| `train_bank_v2` | `runs/banks-v2/banks/train_bank_v2.json` | `f4574ce133892460bc7183a21226d2d85fa8bed9e752903b536004cc69e891b7` | `8eccbbd9a55d9a4f0bca36e10d17c54fe84d475bafeedd7bfa73bc34401a80e4` |
| `dev_bank_v2` | `runs/banks-v2/banks/dev_bank_v2.json` | `6546c98fd3ab63884ea6806480b9b4ad88756c677db10de9e9f0f32f9e18f024` | `3b6823b3ebb150fdf56e80f1b2ad5f550c6d5cfd4b784c2f1386ff505f71f866` |
| `prov_bank_v3` | `runs/prov-v3/banks/prov_bank_v3.json` | `6045dc871150ed9263b21edb0dca031ac60fe2a4443ac8f52e23b8d6fb172948` | `a784fd3d0e0de05077f61c2671acfabe26ec9b1007907621026c43d1425530e4` |

Each of the 30 files `data_test.json` and `data_train.json` in `runs/` records one of these sha256 values.
