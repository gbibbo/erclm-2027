"""LoRA retraining with the recipe of the study under review (qwen-vad-lora scripts/finetune.py),
ported to transformers 5.

Recipe kept from the run of the study under review (run_matrix.py, --skip_vad_filter, seed 42)
    model     Qwen2-Audio-7B-Instruct in NF4 with double quantisation and FP16 compute,
              prepare_model_for_kbit_training, gradient checkpointing (use_reentrant=False)
    LoRA      r 64, alpha 16, dropout 0.05, q_proj k_proj v_proj o_proj, bias none, CAUSAL_LM
    training  3 epochs, batch 2, gradient accumulation 8, lr 5e-5, warmup 100, fp16, seed 42
    prompt    the forced-choice A/B prompt (lib_v2.PROMPTS["ab"]), answer "A" or "B"
    loss      only from the answer token to the end of the sequence
    selection evaluation on the dev bank every 200 steps, final adapter = lowest eval loss.
              transformers 5 also evaluates at the last step, so with 576 steps the candidates
              are steps 200, 400 and 576

Added for unattended runs: a checkpoint every 50 steps is uploaded to the remote and committed with
a COMMIT.json written last. A new Colab or Kaggle session downloads the latest committed checkpoint and resumes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from . import lib_v2 as L
from .storage import md5_file

RECIPE = {"lora_r": 64, "lora_alpha": 16, "lora_dropout": 0.05,
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
          "epochs": 3, "batch_size": 2, "grad_accum": 8, "lr": 5e-5, "warmup_steps": 100, "seed": 42,
          "eval_steps": 200, "checkpoint_every": 50, "prompt_key": "ab"}


class AnswerDataset:
    """Items of a bank manifest as model inputs, with the loss masked up to the answer."""

    def __init__(self, manifest: pd.DataFrame, processor, prompt_key: str = "ab"):
        self.df = manifest.reset_index(drop=True)
        self.processor = processor
        self.prompt = L.PROMPTS[prompt_key]
        self.labels = self.prompt["labels"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = L.load_wav(row["path"])
        answer = self.labels[0] if int(row["label"]) == 1 else self.labels[1]
        conv = [{"role": "system", "content": self.prompt["system"]},
                {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": self.prompt["user"]}]},
                {"role": "assistant", "content": answer}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=False, tokenize=False)
        inputs = L.prepare_inputs(self.processor, text, wav)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = inputs["input_ids"].clone()
        ids = labels.tolist()
        ans = self.processor.tokenizer.encode(answer, add_special_tokens=False)
        start = -1
        for i in range(len(ids) - 1, max(len(ids) - 20, -1), -1):  # the answer sits at the very end
            if ids[i:i + len(ans)] == ans:
                start = i
                break
        if start <= 0:
            raise RuntimeError(f"answer token not found near the end of item {row['variant_id']}")
        labels[:start] = -100
        inputs["labels"] = labels
        return inputs


def collate(batch):
    import torch
    n = max(item["input_ids"].shape[0] for item in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": [], "input_features": [], "feature_attention_mask": []}
    for item in batch:
        pad = n - item["input_ids"].shape[0]
        out["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=0))
        out["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
        out["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad), value=-100))
        out["input_features"].append(item["input_features"])
        out["feature_attention_mask"].append(item["feature_attention_mask"])
    return {k: torch.stack(v) for k, v in out.items()}


def attach_lora(model, recipe: dict = RECIPE, quantized: bool = True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if quantized:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    cfg = LoraConfig(r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"],
                     target_modules=list(recipe["target_modules"]), bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def best_eval_step(log_history: list) -> Optional[int]:
    """Step with the lowest eval_loss, as load_best_model_at_end would pick."""
    evals = [(h["eval_loss"], h["step"]) for h in log_history if "eval_loss" in h]
    return min(evals)[1] if evals else None


# ---------------------------------------------------------------- checkpoints on the remote

def commit_checkpoint(remote, local_dir: Path, rdir: str) -> dict:
    """Upload every file of a checkpoint folder, then COMMIT.json listing them with size and MD5."""
    files = {}
    for p in sorted(local_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(local_dir)).replace("\\", "/")
            files[rel] = remote.put(p, f"{rdir}/{rel}")
    remote.put_json({"files": files}, f"{rdir}/COMMIT.json")
    return files


def latest_committed(remote, rroot: str) -> Optional[str]:
    """Remote folder of the newest checkpoint whose COMMIT.json exists."""
    steps = []
    for name in remote.ls(rroot + "/_index").keys():
        if name.startswith("checkpoint-") and name.endswith(".json"):
            steps.append(int(name[len("checkpoint-"):-5]))
    for step in sorted(steps, reverse=True):
        if remote.exists(f"{rroot}/checkpoint-{step}/COMMIT.json"):
            return f"{rroot}/checkpoint-{step}"
    return None


def fetch_checkpoint(remote, rdir: str, local_dir: Path) -> Path:
    commit = remote.get_json(f"{rdir}/COMMIT.json")
    for rel, info in commit["files"].items():
        target = local_dir / rel
        if not (target.exists() and target.stat().st_size == info["size"] and md5_file(target) == info["md5"]):
            remote.get(f"{rdir}/{rel}", target, expect=info)
    return local_dir


def make_callback(remote, rroot: str, out_dir: Path, keep_steps: set, log: Callable = print,
                  on_step: Optional[Callable] = None):
    """TrainerCallback that commits each saved checkpoint to the remote and prunes old local ones."""
    from transformers import TrainerCallback

    class RemoteCheckpoints(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if on_step:
                on_step(state.global_step, state.max_steps)

        def on_save(self, args, state, control, **kw):
            step = state.global_step
            local = out_dir / f"checkpoint-{step}"
            if not local.is_dir():
                return
            commit_checkpoint(remote, local, f"{rroot}/checkpoint-{step}")
            remote.put_json({"step": step}, f"{rroot}/_index/checkpoint-{step}.json")
            log(f"checkpoint {step} committed to {rroot}")
            saved = sorted(int(p.name.split("-")[1]) for p in out_dir.glob("checkpoint-*") if p.is_dir())
            for s in saved[:-2]:
                if s not in keep_steps:
                    shutil.rmtree(out_dir / f"checkpoint-{s}", ignore_errors=True)
                    if remote.exists(f"{rroot}/checkpoint-{s}/COMMIT.json"):
                        remote.delete(f"{rroot}/_index/checkpoint-{s}.json")
                        remote.purge(f"{rroot}/checkpoint-{s}")

    return RemoteCheckpoints()


def train(model, processor, train_man: pd.DataFrame, dev_man: pd.DataFrame, out_dir, remote, rroot: str,
          recipe: dict = RECIPE, fp16: Optional[bool] = None, log: Callable = print, on_step=None,
          max_steps: int = -1) -> dict:
    """Train, resuming from the newest committed remote checkpoint. Returns a report with the final
    adapter folder (best eval step) and the evaluation history."""
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(recipe["seed"])
    if fp16 is None:
        fp16 = torch.cuda.is_available()
    resume = latest_committed(remote, rroot)
    resume_dir = None
    if resume:
        resume_dir = fetch_checkpoint(remote, resume, out_dir / Path(resume).name)
        log(f"resuming from {resume}")
    args = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=recipe["epochs"], max_steps=max_steps,
        per_device_train_batch_size=recipe["batch_size"], per_device_eval_batch_size=recipe["batch_size"],
        gradient_accumulation_steps=recipe["grad_accum"], warmup_steps=recipe["warmup_steps"],
        learning_rate=recipe["lr"], fp16=fp16, logging_steps=10, save_strategy="steps",
        save_steps=recipe["checkpoint_every"], eval_strategy="steps", eval_steps=recipe["eval_steps"],
        save_total_limit=None, load_best_model_at_end=False, report_to="none", seed=recipe["seed"],
        remove_unused_columns=False, dataloader_num_workers=0)
    keep = set(range(recipe["eval_steps"], 100000, recipe["eval_steps"]))
    trainer = Trainer(model=model, args=args, train_dataset=AnswerDataset(train_man, processor, recipe["prompt_key"]),
                      eval_dataset=AnswerDataset(dev_man, processor, recipe["prompt_key"]), data_collator=collate,
                      callbacks=[make_callback(remote, rroot, out_dir, keep, log, on_step)])
    trainer.train(resume_from_checkpoint=str(resume_dir) if resume_dir else None)
    history = trainer.state.log_history
    best = best_eval_step(history)
    if best is None:  # no evaluation happened (very short runs): keep the last weights
        final = out_dir / "final"
        trainer.save_model(str(final))
        best_from = "last"
    else:
        src = out_dir / f"checkpoint-{best}"
        if not src.is_dir():
            src = fetch_checkpoint(remote, f"{rroot}/checkpoint-{best}", src)
        final = out_dir / "final"
        shutil.rmtree(final, ignore_errors=True)
        final.mkdir(parents=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy2(src / name, final / name)
        best_from = f"checkpoint-{best}"
    evals = [{"step": h["step"], "eval_loss": h["eval_loss"]} for h in history if "eval_loss" in h]
    losses = [{"step": h["step"], "loss": h["loss"]} for h in history if "loss" in h]
    return {"final_dir": str(final), "best_from": best_from, "global_step": trainer.state.global_step,
            "max_steps": trainer.state.max_steps, "evals": evals, "train_loss": losses[-20:],
            "recipe": recipe, "fp16": fp16, "n_train": len(train_man), "n_dev": len(dev_man),
            "resumed_from": resume}


def adapter_summary(final_dir) -> dict:
    files = {p.name: {"size": p.stat().st_size, "md5": md5_file(p)} for p in Path(final_dir).iterdir() if p.is_file()}
    cfg = json.loads((Path(final_dir) / "adapter_config.json").read_text())
    return {"files": files, "r": cfg.get("r"), "lora_alpha": cfg.get("lora_alpha"),
            "target_modules": sorted(cfg.get("target_modules") or [])}

