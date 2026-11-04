# Error in omni-pilot-v1

- Session 20261029-212511-kaggle-a073 on kaggle
- Stage extract_omni
- Time 2026-10-29T21:33:06Z
- Code b583fd7a56564fa48024cb31e4d965bce8dc7139

## Traceback

```
Traceback (most recent call last):
  File "/kaggle/working/erclm-2027/src/erclm/runner.py", line 1347, in run
    self.execute(s)
    ~~~~~~~~~~~~^^^
  File "/kaggle/working/erclm-2027/src/erclm/runner.py", line 832, in execute
    result, files = getattr(self, f"stage_{s.kind}")(s)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
  File "/kaggle/working/erclm-2027/src/erclm/runner.py", line 1210, in stage_omni
    model, proc, info = self.omni_loader()
                        ~~~~~~~~~~~~~~~~^^
  File "/kaggle/working/erclm-2027/src/erclm/runner.py", line 1184, in _load_omni
    return omni.load(self.cfg, self.host.secret("HF_TOKEN", required=False), work=self.work, log=self.log)
           ~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/erclm-2027/src/erclm/omni.py", line 173, in load
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        str(local), config=config, quantization_config=qc, dtype=dt, device_map="auto",
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        max_memory=max_memory).eval()
        ^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/transformers/modeling_utils.py", line 277, in _wrapper
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/modeling_utils.py", line 5048, in from_pretrained
    ) = cls._load_pretrained_model(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~^
        model,
        ^^^^^^
    ...<12 lines>...
        weights_only=weights_only,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/transformers/modeling_utils.py", line 5468, in _load_pretrained_model
    _error_msgs, disk_offload_index = load_shard_file(args)
                                      ~~~~~~~~~~~~~~~^^^^^^
  File "/kaggle/working/erclm-2027/src/erclm/omni.py", line 99, in wrapped
    return load_shard_file(args)
  File "/usr/local/lib/python3.13/dist-packages/transformers/modeling_utils.py", line 843, in load_shard_file
    disk_offload_index = _load_state_dict_into_meta_model(
        model,
    ...<8 lines>...
        device_mesh=device_mesh,
    )
  File "/usr/local/lib/python3.13/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/transformers/modeling_utils.py", line 774, in _load_state_dict_into_meta_model
    hf_quantizer.create_quantized_param(model, param, param_name, param_device)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 233, in create_quantized_param
    new_value = bnb.nn.Params4bit(new_value, requires_grad=False, **kwargs).to(target_device)
  File "/usr/local/lib/python3.13/dist-packages/bitsandbytes/nn/modules.py", line 428, in to
    return self._quantize(device)
           ~~~~~~~~~~~~~~^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/bitsandbytes/nn/modules.py", line 383, in _quantize
    w_4bit, quant_state = bnb.functional.quantize_4bit(
                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        w,
        ^^
    ...<3 lines>...
        quant_storage=self.quant_storage,
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/bitsandbytes/functional.py", line 929, in quantize_4bit
    _out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        A,
        ^^
    ...<2 lines>...
        quant_storage,
        ^^^^^^^^^^^^^^
    )
    ^
  File "/usr/local/lib/python3.13/dist-packages/torch/_ops.py", line 865, in __call__
    return self._op(*args, **kwargs)
           ~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.13/dist-packages/torch/_compile.py", line 54, in inner
    return disable_fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/torch/_dynamo/eval_frame.py", line 1263, in _fn
    return fn(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/torch/library.py", line 751, in func_no_dynamo
    return func(*args, **kwargs)
  File "/usr/local/lib/python3.13/dist-packages/bitsandbytes/backends/cuda/ops.py", line 392, in _
    out = torch.empty(((n + 1) // (quant_storage.itemsize * 2), 1), device=A.device, dtype=quant_storage)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB. GPU 1 has a total capacity of 14.56 GiB of which 2.81 MiB is free. Including non-PyTorch memory, this process has 14.56 GiB memory in use. Of the allocated memory 6.25 GiB is allocated by PyTorch, and 8.19 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

## Entorno

```json
{
  "run_id": "omni-pilot-v1",
  "run_name": "omni-pilot-v1",
  "session_id": "20261029-212511-kaggle-a073",
  "host": "kaggle",
  "started_at": "2026-10-29T21:25:11Z",
  "code_sha": "b583fd7a56564fa48024cb31e4d965bce8dc7139",
  "code_ref": "stable",
  "config_file": "configs/omni_pilot.yaml",
  "config_hash": "6a64e68f65b3db5a",
  "config": {
    "run_name": "omni-pilot-v1",
    "repo": "gbibbo/erclm-2027",
    "remote": "erclm:ERC-LM-2027",
    "model": {
      "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
      "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
      "quant": "nf4"
    },
    "prompt_key": "ab",
    "llm_layers": [
      8,
      16,
      24
    ],
    "backbones": [],
    "adapter": "models/lora_csl_final",
    "data": {
      "test": {
        "tar": "data/test_bank_v2.tar",
        "manifest": "test_bank_v2/manifest.csv",
        "audio": "test_bank_v2/audio",
        "expect": {
          "rows": 21340,
          "conditions": 22,
          "base_ids": 970,
          "rec_ids": 695,
          "speech_share": 0.5
        }
      },
      "train": null
    },
    "selection": {
      "pilot": true,
      "n_per_class": 50,
      "seed": 42
    },
    "extraction": {
      "shard_size": 100,
      "n_content_free": 8,
      "n_head_checks": 3,
      "log_every": 50
    },
    "check": {
      "head_tol": 0.01
    },
    "analysis": {
      "bootstrap_B": 2000,
      "seed": 42,
      "h3_margin": 0.02,
      "prevalence_reps": 20
    },
    "heartbeat": {
      "log_every_s": 300,
      "progress_every_s": 1200
    },
    "stages": [
      "data_test",
      "extract_omni"
    ],
    "audio_transform": null,
    "lora_keep": null,
    "pip": [
      "transformers==4.57.6",
      "huggingface_hub>=0.34,<1.0",
      "tokenizers>=0.22,<=0.23.0"
    ],
    "omni": {
      "prompts": [
        "ab",
        "ba_speech_second",
        "hand_csl"
      ],
      "conditions": [
        "dur_20ms",
        "dur_40ms",
        "dur_1000ms"
      ],
      "compute_dtype": "float16",
      "max_memory_per_gpu": "13GiB",
      "rule": {
        "validity_prompt": "hand_csl",
        "validity_condition": "dur_1000ms",
        "validity_min_ba": 0.85,
        "full_items": 27160,
        "max_hours": 25,
        "read_prompts": [
          "ab",
          "s_sim"
        ],
        "read_conditions": [
          "dur_20ms",
          "dur_40ms"
        ],
        "representation_auroc_hi": 0.65,
        "criterion_auroc_lo": 0.6,
        "criterion_gap": 0.1
      }
    }
  },
  "model": {
    "id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
    "quant": "nf4"
  },
  "backbones": [],
  "stages": [
    "data_test",
    "extract_omni"
  ],
  "force": [],
  "remote": "rclone:erclm:ERC-LM-2027",
  "work": "/tmp/erclm_work",
  "gpu": {
    "cuda": true,
    "torch": "2.11.0+cu128",
    "cuda_version": "12.8",
    "devices": [
      {
        "name": "Tesla T4",
        "total_gb": 14.56
      },
      {
        "name": "Tesla T4",
        "total_gb": 14.56
      }
    ]
  },
  "versions": {
    "python": "3.13.15",
    "platform": "Linux-6.18.55+-x86_64-with-glibc2.39",
    "torch": "2.11.0+cu128",
    "numpy": "2.1.3",
    "pandas": "2.3.3",
    "scipy": "1.16.3",
    "sklearn": "1.6.1",
    "transformers": "4.57.6",
    "peft": "0.21.2",
    "accelerate": "1.14.0",
    "bitsandbytes": "0.50.2",
    "soundfile": "0.14.0",
    "librosa": "0.11.0",
    "yaml": "6.0.3",
    "rclone": "rclone v1.75.2"
  }
}
```

## Last 200 lines of the log

```
[21:25:11] session 20261029-212511-kaggle-a073 on kaggle, run omni-pilot-v1, code b583fd7a5656, remote rclone:erclm:ERC-LM-2027
[21:25:11] plan: data_test, extract_omni
[21:25:59] [data_test] already done on 2026-10-29T21:15:11Z by 20261029-211251-kaggle-9b1e, skipping
[21:26:11] downloading data/test_bank_v2.tar
[21:26:57] extracting test_bank_v2.tar
[21:27:15] work/omni-pilot-v1/omni/ab: 0 committed shards, 0 dropped, 0 orphans removed
[21:27:17] work/omni-pilot-v1/omni/ba_speech_second: 0 committed shards, 0 dropped, 0 orphans removed
[21:27:19] work/omni-pilot-v1/omni/hand_csl: 0 committed shards, 0 dropped, 0 orphans removed
[21:27:19] transformers 4.57.6, disk {'/tmp/erclm_work': {'free_gb': 1122.7, 'total_gb': 8758.4}, '/tmp': {'free_gb': 1122.7, 'total_gb': 8758.4}, '/root': {'free_gb': 1122.7, 'total_gb': 8758.4}}
Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'interleaved', 'mrope_section', 'mrope_interleaved'}
Unrecognized keys in `rope_scaling` for 'rope_type'='default': {'interleaved', 'mrope_section'}

Loading checkpoint shards:   0%|          | 0/13 [00:00<?, ?it/s][21:28:13] shard model-00001-of-00015.safetensors: 5.00 GB fetched and loaded in 33 s

Loading checkpoint shards:   8%|7         | 1/13 [00:33<06:36, 33.07s/it][21:28:44] shard model-00002-of-00015.safetensors: 5.00 GB fetched and loaded in 30 s

Loading checkpoint shards:  15%|#5        | 2/13 [01:03<05:46, 31.46s/it][21:29:13] shard model-00003-of-00015.safetensors: 5.00 GB fetched and loaded in 29 s

Loading checkpoint shards:  23%|##3       | 3/13 [01:32<05:04, 30.44s/it][21:29:42] shard model-00004-of-00015.safetensors: 5.00 GB fetched and loaded in 29 s

Loading checkpoint shards:  31%|###       | 4/13 [02:01<04:27, 29.76s/it][21:30:09] shard model-00005-of-00015.safetensors: 5.00 GB fetched and loaded in 28 s

Loading checkpoint shards:  38%|###8      | 5/13 [02:29<03:52, 29.10s/it][21:30:42] shard model-00006-of-00015.safetensors: 5.00 GB fetched and loaded in 32 s

Loading checkpoint shards:  46%|####6     | 6/13 [03:01<03:31, 30.25s/it][21:31:10] shard model-00007-of-00015.safetensors: 5.00 GB fetched and loaded in 28 s

Loading checkpoint shards:  54%|#####3    | 7/13 [03:29<02:57, 29.52s/it][21:31:40] shard model-00008-of-00015.safetensors: 5.00 GB fetched and loaded in 30 s

Loading checkpoint shards:  62%|######1   | 8/13 [03:59<02:27, 29.55s/it][21:32:09] shard model-00009-of-00015.safetensors: 5.00 GB fetched and loaded in 29 s

Loading checkpoint shards:  69%|######9   | 9/13 [04:28<01:57, 29.47s/it][21:32:39] shard model-00010-of-00015.safetensors: 5.00 GB fetched and loaded in 30 s

Loading checkpoint shards:  77%|#######6  | 10/13 [04:58<01:28, 29.64s/it][21:33:06] shard model-00011-of-00015.safetensors: 5.00 GB fetched and loaded in 27 s

Loading checkpoint shards:  77%|#######6  | 10/13 [05:25<01:37, 32.60s/it]
```
