"""Figure 1 of the paper from the condition tables of the full run (run records in runs/).

    python tools/make_figures.py                                  # writes figures/fig_curves.pdf
    python tools/make_figures.py --out figures/fig_curves.pdf

Needs matplotlib, which is not in requirements.txt because the runs on Colab and Kaggle do not
plot. BA (top) and AUROC (bottom) against duration and SNR. Colour is the readout and line style
the backbone, so the figure also reads in grey. L1 has the AUROC of L0 by construction and is left
out of the bottom row. The palette is blue, orange and aqua on white, with markers and line style as
the secondary encoding. The figures are written to figures/, which is created if missing.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"  # the run records shipped with the repository
RUN = {"base": "full-v1-base/results/base/condition_table.csv",
       "lora": "full-v1-lora/results/lora/condition_table.csv"}
READOUTS = [("L0", "L0", "#2a78d6", "o"), ("L1_bc_cond", "L1 (BC)", "#eb6834", "s"),
            ("L2", "L2", "#1baf7a", "^"), ("P_enc_region_mean", "Encoder probe", "#555555", "D")]
BACKBONES = [("base", "Base", "-"), ("lora", "LoRA", "--")]


def load() -> dict:
    return {b: pd.read_csv(RUNS / path) for b, path in RUN.items()}


def plot(tables: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "pdf.fonttype": 42, "font.family": "sans-serif"})
    fig, axes = plt.subplots(2, 2, figsize=(3.39, 2.85), sharey="row", sharex="col")
    for row, metric in enumerate(("ba", "auroc")):
        for col, axis in enumerate(("duration", "snr")):
            ax = axes[row, col]
            for b, _, style in BACKBONES:
                df = tables[b]
                for key, _, color, marker in READOUTS:
                    if metric == "ba" and key.startswith("P_enc"):
                        continue
                    if metric == "auroc" and key.startswith("L1"):
                        continue
                    d = df[(df.axis == axis) & (df.readout == key)].sort_values("level")
                    ax.plot(d.level, d[metric], linestyle=style, color=color, marker=marker, markersize=2.6,
                            linewidth=0.9, markeredgewidth=0, clip_on=False)
            if axis == "duration":
                ax.set_xscale("log", base=2)
                ax.set_xticks([20, 50, 100, 200, 1000])
                ax.set_xticklabels(["20", "50", "100", "200", "1000"])
                ax.minorticks_off()
            else:
                ax.set_xticks([-10, -5, 0, 5, 10, 20])
            ax.grid(True, linewidth=0.3, color="#d9d9d6")
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 1:
                ax.set_xlabel("Duration (ms)" if axis == "duration" else "SNR (dB)")
        axes[row, 0].set_ylabel("BA" if metric == "ba" else "AUROC")
    axes[0, 0].set_ylim(0.45, 1.0)
    axes[1, 0].set_ylim(0.6, 1.0)
    handles = [Line2D([], [], color=c, marker=m, markersize=3, linewidth=0.9, markeredgewidth=0, label=lab)
               for _, lab, c, m in READOUTS]
    handles += [Line2D([], [], color="#333333", linestyle=s, linewidth=0.9, label=lab) for _, lab, s in BACKBONES]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6.5, handlelength=2.2,
               columnspacing=1.0, bbox_to_anchor=(0.53, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=0.4, w_pad=0.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#555555"
WORK = ROOT / "_work" / "closing"
DURS = (20, 40, 60, 80, 100, 200, 500, 1000)


def _style(plt):
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                         "pdf.fonttype": 42, "font.family": "sans-serif", "legend.fontsize": 6.2})


def _clean(ax):
    ax.grid(True, linewidth=0.3, color="#d9d9d6")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_criterion(ax):
    """(a) Mean native score per class and condition of the base model, with the frozen L1_tr
    threshold and the per-condition batch-calibration thresholds. Exploratory."""
    sc = pd.read_csv(RUNS / "full-v1-base/results/base/scores.csv.gz", compression="gzip")
    conds = [f"dur_{d}ms" for d in DURS] + [f"snr_{s}dB" for s in (-10, -5, 0, 5, 10, 20)]
    xs = list(range(8)) + list(range(9, 15))
    labels = [str(d) for d in DURS] + ["-10", "-5", "0", "5", "10", "20"]
    for lab, color, marker, name in ((1, BLUE, "o", "speech"), (0, ORANGE, "s", "non-speech")):
        g = sc[sc["label"] == lab].groupby("condition")["s_primary"]
        mean, q1, q3 = g.mean().reindex(conds), g.quantile(0.25).reindex(conds), g.quantile(0.75).reindex(conds)
        for seg in (slice(0, 8), slice(8, 14)):
            x = xs[seg]
            ax.fill_between(x, q1.values[seg], q3.values[seg], color=color, alpha=0.18, linewidth=0)
            ax.plot(x, mean.values[seg], color=color, marker=marker, markersize=2.6, linewidth=0.9,
                    markeredgewidth=0, label=name if seg.start == 0 else None)
    bc = sc.groupby("condition")["L1_bc_cond__thr"].first().reindex(conds)
    ax.plot(xs, bc.values, linestyle="none", marker="_", markersize=5, color="#222222", markeredgewidth=1.0,
            label="L1 threshold per condition")
    ax.axhline(-3.137, color="#222222", linestyle="--", linewidth=0.8, label="frozen L1$_{tr}$ threshold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xlim(-0.6, 14.6)
    ax.set_ylim(-5.15, -1.35)
    ax.set_ylabel("Mean native score s (base)")
    ax.text(3.5, 1.02, "duration (ms)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.text(11.5, 1.02, "SNR (dB)", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=6.5)
    ax.annotate("native threshold s = 0\nlies above the plot", xy=(13.6, -1.36), xytext=(13.6, -1.75), fontsize=5.8,
                ha="right", va="center", arrowprops={"arrowstyle": "->", "linewidth": 0.6, "color": "#222222"})
    _clean(ax)
    ax.legend(loc="lower center", frameon=False, handlelength=1.6, ncol=2, columnspacing=0.8)


DATA_SUFFIX = ""  # set by --data-suffix


def panel_regularisation(ax):
    """(b) Base L2 minus LoRA L0 across the pre-registered C grid. Exploratory."""
    c = pd.read_csv(WORK / f"csweep_contrasts{DATA_SUFFIX}.csv")
    c = c[c["what"].str.contains(" - lora L0")].copy()
    c["metric"] = np.where(c["what"].str.startswith("AUROC"), "AUROC", "BA")
    c["C"] = c["what"].str.extract(r"C=([0-9.e-]+)")[0].astype(float)
    series = [("AUROC", "main", "duration", BLUE, "o", "-", "AUROC, duration"),
              ("AUROC", "main", "snr", ORANGE, "s", "-", "AUROC, SNR"),
              ("AUROC", "prov", "duration", AQUA, "^", "-", "AUROC, provenance"),
              ("BA", "main", "duration", BLUE, "o", "--", "BA, duration"),
              ("BA", "main", "snr", ORANGE, "s", "--", "BA, SNR")]
    for met, bank, axis, color, marker, ls, name in series:
        d = c[(c["metric"] == met) & (c["bank"] == bank) & (c["axis"] == axis)].sort_values("C")
        fill = color if ls == "-" else "white"
        ax.errorbar(d["C"], d["estimate"], yerr=[d["estimate"] - d["lo"], d["hi"] - d["estimate"]], color=color,
                    linestyle=ls, marker=marker, markersize=3, markerfacecolor=fill, markeredgewidth=0.7,
                    linewidth=0.9, elinewidth=0.6, capsize=1.2, label=name)
    ax.axhline(-2, color=GREY, linestyle=":", linewidth=0.8, label="H3 margin (BA)")
    ax.axhline(0, color="#999999", linewidth=0.5)
    ax.axvline(1e-2, color=GREY, linestyle="-", linewidth=0.4)
    ax.text(1e-2, 2.6, "chosen by CV", fontsize=5.8, ha="center", va="bottom", color=GREY)
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
    ax.set_xlim(5e-5, 2.2e-1)
    ax.set_xlabel("C (smaller is a stronger penalty)")
    ax.set_ylabel("Base L2 minus LoRA L0")
    _clean(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=2.0, fontsize=5.6, labelspacing=0.25,
              borderaxespad=0.1)


def panel_provenance(ax):
    """(c) Paired AUROC difference from LoRA L0 per duration, main bank against provenance bank. Exploratory."""
    f = pd.read_csv(WORK / f"figure_auroc_diff{DATA_SUFFIX}.csv")
    f["dur"] = f["contrast"].str.extract(r"dur_(\d+)ms")[0].astype(int)
    series = [("FIG main", "base L2cv", BLUE, "o", "-", "L2$_{cv}$, main bank"),
              ("FIG prov", "in-bank L2cv", AQUA, "^", "-", "L2$_{cv}$, provenance bank"),
              ("FIG prov", "transferred", GREY, "s", "--", "transferred head, provenance")]
    for prefix, key, color, marker, ls, name in series:
        d = f[f["contrast"].str.startswith(prefix) & f["contrast"].str.contains(key)].sort_values("dur")
        ax.fill_between(d["dur"], d["lo"], d["hi"], color=color, alpha=0.15, linewidth=0)
        ax.plot(d["dur"], d["estimate"], color=color, marker=marker, markersize=3, linestyle=ls, linewidth=0.9,
                markeredgewidth=0, label=name)
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([20, 50, 100, 200, 1000])
    ax.set_xticklabels(["20", "50", "100", "200", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("AUROC minus LoRA L0")
    _clean(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, fontsize=5.8, borderaxespad=0.1)


def panel_supervision(ax):
    """(Draft v5, b) Per duration, AUROC of heads minus LoRA L0 of the same bank. Main bank with every excerpt
    of the fold and with 182, provenance bank without the recordings heard as speech, refitted inside it
    and transferred from the main bank (tools/intervention_analysis.py figure). Exploratory."""
    f = pd.read_csv(WORK / "figure_v5_supervision.csv")
    series = [("main, every excerpt", BLUE, "o", "-", "main bank, 776 excerpts per fold"),
              ("main, 122 excerpts", BLUE, "o", "--", "main bank, 122 excerpts per fold"),
              ("provenance, refitted inside", AQUA, "^", "-", "provenance, fitted inside"),
              ("provenance, transferred", GREY, "s", "--", "provenance, transferred")]
    for key, color, marker, ls, name in series:
        d = f[f["series"] == key].sort_values("dur")
        ax.fill_between(d["dur"], d["lo"], d["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(d["dur"], d["estimate"], color=color, marker=marker, markersize=3, linestyle=ls, linewidth=0.9,
                markeredgewidth=0, markerfacecolor=color if ls == "-" else "white", label=name)
    ax.axhline(0, color="#999999", linewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks([20, 50, 100, 200, 1000])
    ax.set_xticklabels(["20", "50", "100", "200", "1000"])
    ax.minorticks_off()
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Head AUROC minus LoRA L0")
    _clean(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, fontsize=5.6, borderaxespad=0.1)


def plot_v4(out: Path, panels: str = "abc", labels: str = ""):
    """Panels a (criterion), b (regularisation) and c (provenance), lettered with `labels` in the
    order drawn (default: the panel keys). The paper prints b and c as (a) and (b) and a as (c)."""
    labels = labels or panels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    figsize = {3: (7.0, 2.15), 2: (3.39, 3.35), 1: (3.39, 2.1)}[len(panels)]
    if len(panels) == 3:
        fig, axes = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": [1.25, 1, 1]})
    else:
        fig, axes = plt.subplots(len(panels), 1, figsize=figsize)
    draw = {"a": panel_criterion, "b": panel_regularisation, "c": panel_provenance,
            "s": panel_supervision}
    for ax, p, lab in zip(np.atleast_1d(axes), panels, labels):
        draw[p](ax)
        ax.text(0.0, 1.04, f"({lab})", transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom",
                ha="left")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def plot_v6(out: Path):
    """Figure of draft v6. BA of the training-free recipe on the 470 held-out clips against the speech share of
    the unlabeled calibration batch, one panel per axis, with batch calibration (L1) and the
    two-component mixture (P1) of rule P, and LoRA L0 for reference. Bands span the 2.5 and 97.5
    percentiles of the 200 batches of 150 clips (tools/operating_point.py p --sweep-csv). Exploratory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    sw = pd.read_csv(WORK / "interv" / "P_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.39, 1.75), sharey=False)
    for ax, axis, title in zip(axes, ("duration", "snr"), ("(a) duration, clean", "(b) SNR, white noise")):
        for method, color, marker, ls, name in (("L1", ORANGE, "s", "--", "L1, batch calibration"),
                                                ("P1", BLUE, "o", "-", "P1, two-component mixture")):
            d = sw[(sw["axis"] == axis) & (sw["method"] == method)].sort_values("speech share")
            x = 100 * d["speech share"]
            ax.fill_between(x, d["2.5%"], d["97.5%"], color=color, alpha=0.18, linewidth=0)
            ax.plot(x, d["BA"], color=color, marker=marker, markersize=3, linestyle=ls, linewidth=0.9,
                    markeredgewidth=0, label=name)
        ref = (sw[(sw["axis"] == axis) & (sw["method"] == "P1")]["BA"] - sw[(sw["axis"] == axis) & (sw["method"] == "P1")]["BA - LoRA L0"]).iloc[0]
        ax.axhline(ref, color=GREY, linestyle=":", linewidth=0.9, label="LoRA L0")
        ax.set_xticks([10, 30, 50, 70, 90])
        ax.set_xlabel("Speech in the batch (%)")
        ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left")
        _clean(ax)
    axes[0].set_ylabel("BA (%)")
    axes[0].set_ylim(74, 90)
    axes[1].set_ylim(76, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.0, fontsize=5.6,
               columnspacing=0.9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


V6_CROSSING_LABELS = ("LoRA, calibrated alike (median)", "LoRA, native decision", "base model, recipe")
# Draft v7 names the adapted model "fine-tuned model".
V7_CROSSING_LABELS = ("fine-tuned, calibrated alike (median)", "fine-tuned, native decision", "base, recipe")


def plot_v6_crossing(out: Path, labels=V6_CROSSING_LABELS, height=1.8):
    """Fig. 1 of draft v6. Per-condition BA on the 470 held-out clips of the recipe (L1), of the LoRA's native
    decision (L0) and of the LoRA thresholded with the batch median of its logits, with 95% bands
    from the paired bootstrap (tools/operating_point.py crossing --sweep-csv). Exploratory.
    `labels` names the three series in the legend, in that order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    d = pd.read_csv(WORK / "interv" / "crossing.csv")
    d["axis"] = np.where(d["condition"].str.startswith("dur"), "duration", "snr")
    d["level"] = d["condition"].str.extract(r"_(-?\d+)")[0].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(3.39, height), sharey=True)
    series = [("LoRA median", BLUE, "o", "-", labels[0]),
              ("LoRA L0", BLUE, "o", "--", labels[1]),
              ("recipe", ORANGE, "s", "-", labels[2])]
    for ax, axis, title in zip(axes, ("duration", "snr"), ("(a) duration, clean", "(b) SNR, white noise")):
        for key, color, marker, ls, name in series:
            s = d[(d["axis"] == axis) & (d["series"] == key)].sort_values("level")
            ax.fill_between(s["level"], s["lo"], s["hi"], color=color, alpha=0.13, linewidth=0)
            ax.plot(s["level"], s["estimate"], color=color, marker=marker, markersize=2.6, linestyle=ls,
                    linewidth=0.9, markeredgewidth=0.6, markerfacecolor=color if ls == "-" else "white",
                    label=name)
        if axis == "duration":
            ax.set_xscale("log", base=2)
            ax.set_xticks([20, 50, 100, 200, 1000])
            ax.set_xticklabels(["20", "50", "100", "200", "1000"])
            ax.minorticks_off()
            ax.set_xlabel("Duration (ms)")
        else:
            ax.set_xticks([-10, -5, 0, 5, 10, 20])
            ax.set_xlabel("SNR (dB)")
        ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left")
        _clean(ax)
    axes[0].set_ylabel("BA (%)")
    axes[0].set_ylim(55, 101)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, handlelength=2.0, fontsize=5.6,
               columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def plot_v11(tables: dict, out: Path, height=1.75, all_markers=False, dur_ticks=(20, 50, 100, 200, 1000),
             legend_size=5.4, native_label="native decision (L0)", legend_top=0.9, handlelength=1.8,
             columnspacing=0.7, handletextpad=0.8):
    """Fig. 1 of draft v11. Per-condition BA of the main bank for both models with the native decision (L0),
    with L1 and with the best threshold of each condition (oracle_ba, which uses the test labels).
    Colour is the model and line style the rule, so the figure also reads in grey. The legend of
    draft v14 is larger, with the shorter native label of the table and shorter handles, so that it
    still fits in one row and the figure is not scaled down to the column width."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    _style(plt)
    rules = [("L0", "ba", "--", "white", native_label), ("L1_bc_cond", "ba", "-", None, "L1"),
             ("L0", "oracle_ba", ":", None, "best threshold")]
    models = [("base", BLUE, "o", "base"), ("lora", ORANGE, "s", "fine-tuned")]
    fig, axes = plt.subplots(1, 2, figsize=(3.39, height), sharey=True)
    for ax, axis, title in zip(axes, ("duration", "snr"), ("(a) duration, clean", "(b) SNR, white noise")):
        for b, color, marker, _ in models:
            t = tables[b]
            for readout, col, ls, face, _ in rules:
                s = t[(t["axis"] == axis) & (t["readout"] == readout)].sort_values("level")
                dotted = ls == ":"
                ax.plot(s["level"], 100 * s[col], color=color, linestyle=ls, linewidth=0.9,
                        marker=marker if (all_markers or not dotted) else None,
                        markersize=1.8 if dotted else 2.6, markeredgewidth=0.6, markerfacecolor=face or color)
        if axis == "duration":
            ax.set_xscale("log", base=2)
            ax.set_xticks(list(dur_ticks))
            ax.set_xticklabels([str(d) for d in dur_ticks])
            ax.minorticks_off()
            ax.set_xlabel("Duration (ms)")
        else:
            ax.set_xticks([-10, -5, 0, 5, 10, 20])
            ax.set_xlabel("SNR (dB)")
        ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left")
        _clean(ax)
    axes[0].set_ylabel("BA (%)")
    axes[0].set_ylim(45, 101)
    handles = [Line2D([], [], color=c, marker=m, markersize=3, linewidth=0.9, markeredgewidth=0, label=lab)
               for _, c, m, lab in models]
    handles += [Line2D([], [], color="#333333", linestyle=ls, linewidth=0.9, label=lab) for *_, ls, _, lab in rules]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, handlelength=handlelength, fontsize=legend_size,
               columnspacing=columnspacing, handletextpad=handletextpad, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, legend_top), w_pad=0.8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "figures" / "fig_curves.pdf"))
    ap.add_argument("--v4", default="", help="panels of the figure of draft v4, e.g. abc or bc; writes figures/fig_v4_<panels>.pdf")
    ap.add_argument("--labels", default="", help="letters printed on the panels, e.g. ab for --v4 bc")
    ap.add_argument("--v5", default="", help="panels of the figure of draft v5, e.g. bs; writes figures/fig_v5_<panels>.pdf")
    ap.add_argument("--data-suffix", default="", help="suffix of the data files, e.g. _prov-v3-clean")
    ap.add_argument("--v6", action="store_true", help="the prevalence sweep of draft v6, figures/fig_v6_sweep.pdf")
    ap.add_argument("--v7", action="store_true", help="the crossing figure of draft v7, figures/fig_v7_crossing.pdf")
    ap.add_argument("--v11", action="store_true", help="the criterion figure of draft v11, figures/fig_v11_criterion.pdf")
    ap.add_argument("--v12", action="store_true", help="the same figure, shorter, for draft v12, figures/fig_v12_criterion.pdf")
    ap.add_argument("--v13", action="store_true",
                    help="the draft v12 figure with markers on every curve and the evaluated durations as ticks, figures/fig_v13_criterion.pdf")
    ap.add_argument("--v14", action="store_true",
                    help="the draft v13 figure at the draft v12 height with a larger legend, figures/fig_v14_criterion.pdf")
    args = ap.parse_args(argv)
    global DATA_SUFFIX
    DATA_SUFFIX = args.data_suffix
    if args.v14:
        print("wrote", plot_v11(load(), ROOT / "figures" / "fig_v14_criterion.pdf", height=1.5,
                                all_markers=True, dur_ticks=(20, 40, 100, 200, 1000), legend_size=6.5,
                                native_label="native (L0)", legend_top=0.9, handlelength=1.5,
                                columnspacing=0.6, handletextpad=0.4))
    elif args.v13:
        print("wrote", plot_v11(load(), ROOT / "figures" / "fig_v13_criterion.pdf", height=1.38,
                                all_markers=True, dur_ticks=(20, 40, 100, 200, 1000)))
    elif args.v12:
        print("wrote", plot_v11(load(), ROOT / "figures" / "fig_v12_criterion.pdf", height=1.5))
    elif args.v11:
        print("wrote", plot_v11(load(), ROOT / "figures" / "fig_v11_criterion.pdf"))
    elif args.v7:
        print("wrote", plot_v6_crossing(ROOT / "figures" / "fig_v7_crossing.pdf", V7_CROSSING_LABELS, 1.55))
    elif args.v6:
        print("wrote", plot_v6(ROOT / "figures" / "fig_v6_sweep.pdf"))
        print("wrote", plot_v6_crossing(ROOT / "figures" / "fig_v6_crossing.pdf"))
    elif args.v5:
        print("wrote", plot_v4(ROOT / "figures" / f"fig_v5_{args.v5}.pdf", args.v5, args.labels))
    elif args.v4:
        print("wrote", plot_v4(ROOT / "figures" / f"fig_v4_{args.v4}.pdf", args.v4, args.labels))
    else:
        print("wrote", plot(load(), Path(args.out)))


if __name__ == "__main__":
    main()
