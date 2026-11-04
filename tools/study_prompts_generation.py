"""The registered reading of the study's prompts read by generation (2 November 2026).

The rule is in docs/exploratory_log.md, "Generation readout of the study prompts (2 November)",
written before the runs gen-<p>. For each backbone and prompt p of the study, on the 4,200
items of the 300 control clips, averaged over each axis,

    agreement   share of items where the parsed answer and the first-token rule (s > 0) agree
    BA_text     BA of the parsed answer, UNKNOWN counted as not speech, as the study did
    G_text      BA_text(p) - BA_L0(ab)
    L1ab_text   BA_L1(ab) - BA_text(p), and the recipe (s_sim with L1) - BA_text(p)

and, descriptive, a ranking score built from the first tokens the model writes.

Everything is computed twice. "registered" parses with normalize_to_binary, as the rule said.
"qwen2_parser" parses with the study's Qwen2-Audio parser (genread.parse_qwen2_audio), added on
2 November as a declared secondary reading after normalize_to_binary turned out to read "This is
not human speech" as speech. That parser reproduces 21,339 of the 21,340 study predictions.

    python tools/study_prompts_generation.py <root> [--B 2000] [--out <folder>] [--partial]

--partial reads the backbone and prompt pairs whose gen_<p>_<b> folder exists and lists the
others under "missing". Every quantity of a pair depends only on that pair, the references and
the shared bootstrap draw, so a pair read early gives the same numbers as in the full reading.

<root> holds, as the runner writes them,

    q2_base/, q2_lora/        work/full-v1/<backbone>/eval/                (ab)
    q2_base_ba/, q2_lora_ba/  work/prompt-ba_speech_second/<backbone>/eval/
    gen_<p>_base/, gen_<p>_lora/   work/gen-<p>/<backbone>/eval/   (CSV and npz)

Intervals are 95% percentiles of a bootstrap over recordings stratified by class, 2,000
replicates, seed 42, one draw shared by every readout. L1 is re-estimated in every replicate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omni_control as oc  # noqa: E402
from erclm import genread  # noqa: E402
from erclm import lib_v2 as L  # noqa: E402

PROMPTS = ["opro_llm_base", "opro_tmpl", "hand_csl"]
PRIMARY = {"base": "opro_llm_base", "lora": "opro_tmpl"}
PARSERS = {"registered": "gen_label", "qwen2_parser": "gen_label_q2"}
AXES = {"duration": oc.DUR, "snr": oc.SNR, "short": oc.SHORT}
S_TOKEN = 50  # "S" in the Qwen2-Audio tokenizer, the first token the fine-tuned model writes on many non-speech clips


def token_sets(meta: pd.DataFrame, top_ids: np.ndarray, min_n: int = 5, purity: float = 0.9,
               col: str = "gen_label") -> tuple:
    """First generated tokens whose answers parse as speech, or as not speech, in at least `purity`
    of their at least `min_n` occurrences. Built from the parsed text, never from the labels."""
    first = pd.Series(top_ids[:, 0], index=meta.index)
    sp, ns = [], []
    for t, g in meta.groupby(first):
        if len(g) < min_n:
            continue
        share = (g[col] == "SPEECH").mean()
        if share >= purity:
            sp.append(int(t))
        elif (g[col] == "NONSPEECH").mean() >= purity:
            ns.append(int(t))
    return sp, ns


def gen_score(top_ids: np.ndarray, top_logits: np.ndarray, sp: list, ns: list) -> np.ndarray:
    """logsumexp of the first-step logits of the speech tokens minus that of the non-speech tokens.
    A token outside an item's top 20 counts at the 20th logit minus 1, a bound from above."""
    floor = top_logits[:, -1:] - 1.0

    def lse(tokens):
        z = np.stack([np.where(top_ids == t, top_logits, -np.inf).max(1) for t in tokens], 1)
        z = np.where(np.isfinite(z), z, floor)
        m = z.max(1, keepdims=True)
        return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
    return lse(sp) - lse(ns)


def main(root: Path, B: int = 2000, seed: int = 42, out: Path = None, partial: bool = False):
    root = Path(root)
    out = Path(out or root / "reading_generated")
    out.mkdir(parents=True, exist_ok=True)
    ref, _ = oc.load(root / "q2_base_ba")
    ref = ref[ref["condition"].isin(oc.DUR + oc.SNR)]
    ids, meta = list(ref.index), ref
    y = meta["label"].to_numpy().astype(bool)
    S, D = {}, {}
    res = {"B": B, "seed": seed, "answers": {}, "token_sets": {}, "S_token": {}, "parser_agreement": {},
           "missing": []}
    for b in ("base", "lora"):
        for k, f in ((f"{b}_ab", f"q2_{b}"), (f"{b}_ba", f"q2_{b}_ba")):
            m, _ = oc.aligned(*oc.load(root / f, ids=set(ids)), ids)
            S[k] = m["s_primary"].to_numpy(float)
        S[f"{b}_sim"] = 0.5 * (S[f"{b}_ab"] + S.pop(f"{b}_ba"))
        for p in PROMPTS:
            if partial and not (root / f"gen_{p}_{b}").is_dir():
                res["missing"].append(f"{b}/{p}")
                continue
            m, a = oc.aligned(*oc.load(root / f"gen_{p}_{b}", ids=set(ids),
                                       features=("first_top_ids", "first_top_logits")), ids)
            if not (m["label"].to_numpy() == meta["label"].to_numpy()).all():
                raise SystemExit(f"labels of gen_{p}_{b} differ")
            m["answer"] = m["answer"].fillna("").astype(str)
            m["gen_label_q2"] = m["answer"].map(genread.parse_qwen2_audio)
            res["parser_agreement"][f"{b}/{p}"] = float((m["gen_label"] == m["gen_label_q2"]).mean())
            ti, tl = a["first_top_ids"].astype(int), a["first_top_logits"].astype(float)
            S[f"{b}_{p}_first"] = m["s_primary"].to_numpy(float)
            first_rule = m["s_primary"].to_numpy() > 0
            res["answers"][f"{b}/{p}"] = {lab: m.loc[m["label"] == lab, "answer"].str.slice(0, 40)
                                          .value_counts().head(6).to_dict() for lab in (1, 0)}
            for pk, col in PARSERS.items():
                sp, ns = token_sets(m, ti, col=col)
                res["token_sets"][f"{pk}/{b}/{p}"] = {"speech": sp, "nonspeech": ns}
                S[f"{b}_{p}_gen_{pk}"] = gen_score(ti, tl, sp, ns) if sp and ns else np.zeros(len(m))
                D[f"{b}_{p}_{pk}"] = (m[col] == "SPEECH").to_numpy()
                s_rows = (ti[:, 0] == S_TOKEN) & (m["label"].to_numpy() == 0)
                res["S_token"][f"{pk}/{b}/{p}"] = {
                    "n_nonspeech_starting_S": int(s_rows.sum()),
                    "share_parsed_nonspeech": float((m.loc[s_rows, col] == "NONSPEECH").mean()) if s_rows.any() else None,
                    "answers": m.loc[s_rows, "answer"].value_counts().head(5).to_dict()}
                agree = D[f"{b}_{p}_{pk}"] == first_rule
                res[f"{pk}/{b}/{p}/agreement"] = {ax: float(agree[meta["condition"].isin(cs).to_numpy()].mean())
                                                  for ax, cs in AXES.items()}

    conds = list(pd.unique(meta["condition"]))
    masks = {c: (meta["condition"] == c).to_numpy() for c in conds}

    def evaluate(w):
        r = {k: oc.metrics(meta, s, w) for k, s in S.items()}
        for k, d in D.items():
            r[f"{k}_text"] = {c: {"BA": L.ba_w(y[cm], d[cm], None if w is None else w[cm])} for c, cm in masks.items()}
        return r

    point = evaluate(None)
    reps = [evaluate(w) for w in oc.boot_weights(meta["rec_id"].astype(str).to_numpy(), y, B, seed)]

    def q(r, pk, b, p, cs):
        T = oc.summarise(r[f"{b}_{p}_{pk}_text"], "BA", cs)
        G = T - oc.summarise(r[f"{b}_ab"], "L0", cs)
        gen = f"{b}_{p}_gen_{pk}"
        O = oc.summarise(r[gen], "oracle", cs) - oc.summarise(r[f"{b}_ab"], "oracle", cs)
        return {"BA_text": T, "BA_L0_first": oc.summarise(r[f"{b}_{p}_first"], "L0", cs), "G_text": G,
                "O_gen": O, "C_gen": G - O,
                "dAUROC_gen": oc.summarise(r[gen], "auroc", cs) - oc.summarise(r[f"{b}_ab"], "auroc", cs),
                "L1ab_minus_text": oc.summarise(r[f"{b}_ab"], "L1", cs) - T,
                "recipe_minus_text": oc.summarise(r[f"{b}_sim"], "L1", cs) - T}

    rows = []
    for pk in PARSERS:
        for b in ("base", "lora"):
            for p in PROMPTS:
                if f"{b}/{p}" in res["missing"]:
                    continue
                for ax, cs in AXES.items():
                    pt, rd = q(point, pk, b, p, cs), [q(r, pk, b, p, cs) for r in reps]
                    d = {k: {"value": float(v), "ci": [float(np.nanpercentile([x[k] for x in rd], 2.5)),
                                                       float(np.nanpercentile([x[k] for x in rd], 97.5))]}
                         for k, v in pt.items()}
                    d["first_token_misreads"] = bool(1 - res[f"{pk}/{b}/{p}/agreement"][ax] > 0.10)
                    d["calibration_reaches_prompt"] = bool(d["L1ab_minus_text"]["ci"][0] > -0.02)
                    res[f"{pk}/{b}/{p}/{ax}"] = d
                    rows += [{"parser": pk, "backbone": b, "prompt": p, "axis": ax, "quantity": k,
                              "value": v["value"], "lo": v["ci"][0], "hi": v["ci"][1]}
                             for k, v in d.items() if isinstance(v, dict)]
    res["primary"] = {}
    for pk in PARSERS:
        s_lora = res["S_token"].get(f"{pk}/lora/opro_tmpl", {}).get("share_parsed_nonspeech")
        res["primary"][pk] = {
            **{f"{b}/{PRIMARY[b]}/{ax}": {k: res[f"{pk}/{b}/{PRIMARY[b]}/{ax}"][k]
                                          for k in ("first_token_misreads", "calibration_reaches_prompt")}
               for b in PRIMARY for ax in ("duration", "snr") if f"{b}/{PRIMARY[b]}" not in res["missing"]},
            "lora_S_is_not_speech": None if s_lora is None else bool(s_lora >= 0.5)}
    pd.DataFrame(rows).to_csv(out / "generated.csv", index=False)
    (out / "reading.json").write_text(json.dumps(res, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["primary"], indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    main(Path(a.root), B=a.B, seed=a.seed, out=a.out, partial=a.partial)
