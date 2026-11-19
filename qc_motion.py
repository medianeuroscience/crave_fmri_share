#!/usr/bin/env python
"""
qc_motion.py - motion and data-quality screening for the CRAVE fMRIPrep outputs (v2).

Why filtered FD: with the multiband multi-echo sequence, breathing shows up in the
motion estimates as fake head movement, mostly along the phase-encode axis (trans_y).
FD computed from the raw estimates therefore overstates motion. The six motion
parameters are low-pass filtered (0.1 Hz) before FD is computed (Fair et al., 2020;
Gratton et al., 2020). Because filtering also smooths real sudden jerks, a raw-FD
spike criterion is kept as well.

PROPOSED RULE (applied to every participant; change with the flags below):
  * Censor a volume if filtered FD > 0.3 mm  OR  raw FD > 0.9 mm
  * Keep a run if >= 50% of its volumes survive censoring
    (the >= 3 events per regressor check needs the event files and is not done here)
  * Exclude a participant with fewer than 3 usable VS (game) rounds
The old rule (run excluded if mean raw FD > 0.5 mm or > 20% of volumes > 0.5 mm)
and alternative thresholds are also scored, so you can compare retention across
rules before any group analysis. Choose and lock the rule BEFORE looking at group results.

Outputs (in --out, default derivatives/qc):
  run_qc.csv                one row per participant x run (raw + filtered FD, censoring)
  participant_qc.csv        one row per participant, decision under the proposed rule
  rule_comparison.csv       runs / participants retained under each candidate rule
  censor/<run>_desc-censor_timeseries.tsv   0/1 per volume (1 = censored) for the GLM
  visual_qc_ratings.csv     blank sheet for rating the fMRIPrep HTML reports
  fd_heatmap.png            % of volumes censored per participant x run
  fd_traces/sub-XXX.png     raw and filtered FD over time, censored volumes marked

Usage:
  conda activate neuroconda_dev
  python qc_motion.py --skip                     # all participants, proposed rule
  python qc_motion.py --skip --fd-filt 0.2       # try a stricter filtered threshold
  python qc_motion.py --skip --tsnr --roi NAcc=/path/nacc_2mm.nii.gz
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

EXPECTED = ["VS_run-01", "VS_run-02", "VS_run-03", "VS_run-04",
            "MID_run-01", "MID_run-02", "LB"]
SPACE = "space-MNI152NLin2009cAsym_res-2"
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
FNAME = re.compile(r"(?P<sub>sub-[^_]+)_(?:ses-[^_]+_)?task-(?P<task>[^_]+)"
                   r"(?:_run-(?P<run>\d+))?_desc-confounds_timeseries\.tsv$")

# Colors: sequential blue for magnitude, gray for the raw reference line,
# critical red reserved for "fails a rule / censored"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LINE_BLUE = "#2a78d6"
RAW_GRAY = "#b4b2ab"
CRITICAL = "#d03b3b"
INK, INK2, GRID, SURFACE = "#1f1f1f", "#5f5f5f", "#e4e3df", "#fcfcfb"


# ---------------------------------------------------------------- computations
def power_fd(m):
    """Framewise displacement (Power et al., 2012): rotations in radians -> mm on a 50 mm sphere."""
    d = np.abs(np.diff(m, axis=0))
    d[:, 3:] *= 50.0
    return np.concatenate([[0.0], d.sum(axis=1)])


def lowpass(m, tr, cutoff):
    nyq = 0.5 / tr
    if cutoff >= nyq:
        return m.copy()
    if len(m) < 4:
        return m.copy()
    b, a = butter(2, cutoff / nyq, btype="low")
    return filtfilt(b, a, m, axis=0, padlen=min(3 * max(len(a), len(b)), len(m) - 1))


def get_tr(conf_path, bids_root, default):
    cands = [conf_path.with_name(conf_path.name.replace("desc-confounds_timeseries.tsv",
                                                        SPACE + "_desc-preproc_bold.json"))]
    m = FNAME.search(conf_path.name)
    if m:
        tag = conf_path.name.split("_desc-confounds")[0]
        cands += [Path(p) for p in glob.glob(str(Path(bids_root) / m.group("sub") / "ses-01" / "func" /
                                                 (tag + "_echo-1_bold.json")))]
    for c in cands:
        if c.exists():
            try:
                return float(json.load(open(str(c)))["RepetitionTime"]), "json"
            except (KeyError, ValueError):
                pass
    return default, "default"


def censor_mask(raw, filt, fd_filt, fd_raw):
    return (filt > fd_filt) | (raw > fd_raw)


def legacy_fail(raw):
    v = raw[1:]
    return bool((v.mean() > 0.5) or (100 * (v > 0.5).mean() > 20))


def tsnr_in_masks(bold_path, masks):
    import nibabel as nib
    img = nib.load(str(bold_path))
    data = img.dataobj
    tsnr = np.zeros(img.shape[:3], dtype=np.float32)
    for z in range(img.shape[2]):
        sl = np.asarray(data[:, :, z, :], dtype=np.float32)
        mu, sd = sl.mean(axis=-1), sl.std(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr[:, :, z] = np.where(sd > 0, mu / sd, 0)
    return {k: float(np.median(tsnr[v])) if v.any() else np.nan for k, v in masks.items()}


def load_mask(path, shape):
    import nibabel as nib
    mk = np.asarray(nib.load(str(path)).dataobj) > 0
    if mk.shape != tuple(shape):
        raise ValueError("%s: mask shape %s != BOLD %s; resample the ROI to MNI 2 mm first"
                         % (path, mk.shape, shape))
    return mk


def robust_low(x, k=2.5):
    x = pd.Series(x, dtype=float)
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return (x < med - k * mad) if mad > 0 else pd.Series(False, index=x.index)


# ---------------------------------------------------------------- figures
def plot_traces(sub, tr_data, a, out_dir):
    import matplotlib.pyplot as plt
    labels = [l for l in EXPECTED if l in tr_data] + sorted(set(tr_data) - set(EXPECTED))
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 1.6 * len(labels) + 1.0),
                             sharey=True, squeeze=False, facecolor=SURFACE)
    ymax = max(1.2, a.fd_raw * 1.15)
    for ax, lab in zip(axes[:, 0], labels):
        raw, filt, cens = tr_data[lab]
        x = np.arange(len(raw))
        ax.set_facecolor(SURFACE)
        ax.plot(x, np.clip(raw, 0, ymax), color=RAW_GRAY, lw=0.9, label="Raw FD")
        ax.plot(x, filt, color=LINE_BLUE, lw=1.6, label="Filtered FD (< %.2g Hz)" % a.lp)
        ax.scatter(x[cens], np.full(int(cens.sum()), ymax * 0.97), marker="|", s=40,
                   color=CRITICAL, linewidths=1.2, label="Censored volume", zorder=3)
        ax.axhline(a.fd_filt, color=LINE_BLUE, lw=0.8, ls="--")
        ax.axhline(a.fd_raw, color=INK2, lw=0.8, ls=":")
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, len(raw))
        kept = 100 * (1 - cens.mean())
        status = "kept" if kept >= a.min_retained else "FAILS"
        ax.set_title(lab, loc="left", fontsize=9, color=INK, fontweight="bold", pad=3)
        ax.set_title("%d of %d volumes censored | %.0f%% retained | %s"
                     % (cens.sum(), len(cens), kept, status),
                     loc="right", fontsize=8, color=CRITICAL if status == "FAILS" else INK2, pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[-1, 0].set_xlabel("Volume", color=INK2, fontsize=9)
    fig.text(0.01, 0.5, "Framewise displacement (mm)", rotation=90, va="center", color=INK2, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.99, 0.982))
    fig.suptitle("%s: motion by run (dashed = filtered %.2g mm, dotted = raw %.2g mm)"
                 % (sub, a.fd_filt, a.fd_raw), x=0.01, ha="left", y=0.99, fontsize=10, color=INK)
    fig.tight_layout(rect=(0.03, 0, 1, 0.965))
    fig.savefig(str(out_dir / (sub + ".png")), dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(runs, a, out_path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    piv = runs.pivot(index="sub", columns="run_label", values="pct_censored")
    fail = runs.pivot(index="sub", columns="run_label", values="run_excluded")
    cols = [c for c in EXPECTED if c in piv.columns] + sorted(set(piv.columns) - set(EXPECTED))
    order = piv.mean(axis=1).sort_values(ascending=False).index
    piv, fail = piv.loc[order, cols], fail.loc[order, cols]
    bounds = [0, 5, 10, 20, 30, 40, 50, 100]
    cmap = ListedColormap(BLUE_RAMP)
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 0.28 * len(piv) + 1.8), facecolor=SURFACE)
    vals = piv.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap, norm=norm)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#f0efec", zorder=1))
                ax.text(j, i, "missing", ha="center", va="center", fontsize=7, color=INK2, zorder=2)
                continue
            ax.text(j, i, "%.0f%%" % v, ha="center", va="center", fontsize=7.5,
                    color="white" if v >= 30 else INK, zorder=2)
            if bool(fail.iat[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.42, i - 0.38), 0.84, 0.76, fill=False,
                                           ec=CRITICAL, lw=2.5, zorder=5))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_run-0", " ") for c in cols], fontsize=8, color=INK)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(list(piv.index), fontsize=7.5, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=bounds[:-1])
    cb.set_label("% of volumes censored", color=INK2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Share of volumes censored per run, most-censored participants first\n"
                 "Censor: filtered FD > %.2g mm or raw FD > %.2g mm.  Red outline = < %.0f%% retained (run fails)"
                 % (a.fd_filt, a.fd_raw, a.min_retained), loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=170, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fmriprep", default="/srv/lab/fmri/crave/derivatives/fmriprep")
    ap.add_argument("--bids", default="/srv/lab/fmri/crave/crave_bids")
    ap.add_argument("--out", default="/srv/lab/fmri/crave/derivatives/qc")
    ap.add_argument("--subjects", nargs="*", help="limit to these labels, e.g. sub-048087")
    ap.add_argument("--skip", nargs="*", default=[], help="folders to ignore")
    ap.add_argument("--tr", type=float, default=1.761, help="fallback TR if no sidecar is found")
    ap.add_argument("--lp", type=float, default=0.1, help="low-pass cutoff for motion params (Hz)")
    ap.add_argument("--fd-filt", type=float, default=0.3, help="filtered FD censoring threshold (mm)")
    ap.add_argument("--fd-raw", type=float, default=0.9, help="raw FD spike threshold (mm)")
    ap.add_argument("--min-retained", type=float, default=50.0, help="%% volumes a run must keep")
    ap.add_argument("--min-vs", type=int, default=3, help="usable VS rounds a participant needs")
    ap.add_argument("--tsnr", action="store_true")
    ap.add_argument("--roi", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    root, out = Path(a.fmriprep), Path(a.out)
    (out / "censor").mkdir(parents=True, exist_ok=True)
    rois = dict(r.split("=", 1) for r in a.roi)
    subs = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    if a.subjects:
        subs = [s for s in subs if s in a.subjects]
    subs = [s for s in subs if s not in (a.skip or [])]
    print("Screening %d participants | censor: filtered FD > %.2g mm (low-pass %.2g Hz) or raw FD > %.2g mm"
          % (len(subs), a.fd_filt, a.lp, a.fd_raw))

    rows, store, tr_defaults = [], {}, 0
    for sub in subs:
        files = sorted((root / sub).rglob("*_desc-confounds_timeseries.tsv"))
        if not files:
            print("  %s: no confounds files (fMRIPrep incomplete?)" % sub)
            continue
        store[sub] = {}
        for f in files:
            m = FNAME.search(f.name)
            if not m:
                continue
            lab = "%s_run-%02d" % (m.group("task"), int(m.group("run"))) if m.group("run") else m.group("task")
            df = pd.read_csv(str(f), sep="\t")
            mot = df[MOTION].fillna(0).to_numpy(dtype=float)
            tr, src = get_tr(f, a.bids, a.tr)
            tr_defaults += src == "default"
            if len(df) < 20:
                print("  WARNING %s: only %d volumes - probably an aborted run" % (f.name, len(df)))
            raw = power_fd(mot)
            filt = power_fd(lowpass(mot, tr, a.lp))
            cens = censor_mask(raw, filt, a.fd_filt, a.fd_raw)
            cens[0] = False
            pct_c = 100 * cens.mean()
            dv = df["std_dvars"].to_numpy(dtype=float) if "std_dvars" in df else np.full(len(df), np.nan)
            row = {
                "sub": sub, "task": m.group("task"), "run": m.group("run") or "", "run_label": lab,
                "tr": tr, "n_vols": len(df), "minutes": len(df) * tr / 60,
                "mean_fd_raw": raw[1:].mean(), "median_fd_raw": np.median(raw[1:]), "max_fd_raw": raw.max(),
                "pct_raw_over_05": 100 * (raw[1:] > 0.5).mean(),
                "mean_fd_filt": filt[1:].mean(), "median_fd_filt": np.median(filt[1:]),
                "pct_filt_over_thr": 100 * (filt[1:] > a.fd_filt).mean(),
                "n_raw_spikes": int((raw > a.fd_raw).sum()),
                "n_censored": int(cens.sum()), "pct_censored": pct_c,
                "pct_retained": 100 - pct_c, "minutes_retained": (len(df) - cens.sum()) * tr / 60,
                "mean_std_dvars": np.nanmean(dv),
                "legacy_rule_excluded": legacy_fail(raw),
            }
            if a.tsnr:
                bold = f.with_name(f.name.replace("desc-confounds_timeseries.tsv", SPACE + "_desc-preproc_bold.nii.gz"))
                mask = f.with_name(f.name.replace("desc-confounds_timeseries.tsv", SPACE + "_desc-brain_mask.nii.gz"))
                if bold.exists() and mask.exists():
                    import nibabel as nib
                    shape = nib.load(str(bold)).shape[:3]
                    masks = {"brain": load_mask(mask, shape)}
                    for name, p in rois.items():
                        masks[name] = load_mask(p, shape) & masks["brain"]
                    for name, val in tsnr_in_masks(bold, masks).items():
                        row["tsnr_" + name] = val
            rows.append(row)
            store[sub][lab] = (raw, filt, cens)
            pd.DataFrame({"censored": cens.astype(int), "fd_raw": raw.round(4), "fd_filtered": filt.round(4)}) \
                .to_csv(str(out / "censor" / f.name.replace("desc-confounds", "desc-censor")), sep="\t", index=False)
        print("  %s: %d runs" % (sub, len(store[sub])))
    if tr_defaults:
        print("  NOTE: TR not found in %d sidecars; used --tr %.3f for those runs" % (tr_defaults, a.tr))

    runs = pd.DataFrame(rows)
    if runs.empty:
        raise SystemExit("No confounds files found - check --fmriprep")
    task_median = runs.groupby("task")["n_vols"].median().to_dict()
    runs["incomplete"] = [n < 0.5 * task_median[t] for n, t in zip(runs["n_vols"], runs["task"])]
    runs["run_excluded"] = (runs["pct_retained"] < a.min_retained) | runs["incomplete"]
    reasons = []
    for v, inc, n, t in zip(runs["pct_retained"], runs["incomplete"], runs["n_vols"], runs["task"]):
        r = []
        if inc:
            r.append("incomplete run (%d vols < 50%% of %s median %d)" % (n, t, task_median[t]))
        if v < a.min_retained:
            r.append("%.0f%% retained < %.0f%%" % (v, a.min_retained))
        reasons.append("; ".join(r))
    runs["exclusion_reason"] = reasons
    checks = pd.Series("", index=runs.index)
    for task, g in runs.groupby("task"):
        checks[g.index[(g["n_vols"] < 20).to_numpy()]] += "aborted? (<20 volumes); "
        checks[g.index[(g["n_vols"] < g["n_vols"].median() * 0.9).to_numpy()]] += "short run (<90% of task median); "
        checks[g.index[(g["n_vols"] > g["n_vols"].median() * 1.25).to_numpy()]] += "long run (>125% of task median); "
        for c in [c for c in runs.columns if c.startswith("tsnr_")]:
            checks[g.index[robust_low(g[c]).to_numpy()]] += "low %s tSNR; " % c[5:]
    runs["check_flags"] = checks.str.rstrip("; ")
    runs = runs.sort_values(["sub", "run_label"])
    runs.round(4).to_csv(str(out / "run_qc.csv"), index=False)

    # participant table (proposed rule + old rule for comparison)
    prow = []
    for sub in subs:
        g = runs[runs["sub"] == sub]
        present = set(g["run_label"])
        missing = [r for r in EXPECTED if r not in present]
        vs = g[g["task"] == "VS"]
        vs_kept = int((~vs["run_excluded"]).sum())
        vs_legacy_kept = int((~vs["legacy_rule_excluded"]).sum())
        reason = []
        if g.empty:
            reason.append("no fMRIPrep output")
        elif vs_kept < a.min_vs:
            reason.append("%d usable VS rounds < %d" % (vs_kept, a.min_vs))
        prow.append({
            "sub": sub, "n_runs_found": len(g), "missing_runs": ", ".join(missing),
            "extra_runs": ", ".join(sorted(present - set(EXPECTED))),
            "vs_rounds_kept": vs_kept,
            "mid_runs_kept": int(((g["task"] == "MID") & ~g["run_excluded"]).sum()),
            "lb_kept": int(((g["task"] == "LB") & ~g["run_excluded"]).sum()),
            "mean_pct_censored": g["pct_censored"].mean() if len(g) else np.nan,
            "mean_fd_filt": g["mean_fd_filt"].mean() if len(g) else np.nan,
            "mean_fd_raw": g["mean_fd_raw"].mean() if len(g) else np.nan,
            "runs_excluded": ", ".join(g.loc[g["run_excluded"], "run_label"]),
            "runs_to_check": ", ".join(g.loc[g["check_flags"] != "", "run_label"]),
            "participant_excluded": bool(reason),
            "participant_reason": "; ".join(reason),
            "old_rule_vs_kept": vs_legacy_kept,
            "old_rule_excluded": (not g.empty) and vs_legacy_kept < 3,
        })
    parts = pd.DataFrame(prow)
    parts.round(4).to_csv(str(out / "participant_qc.csv"), index=False)

    # rule comparison (retention only - no outcome data involved)
    candidates = [("Old: raw FD (mean > 0.5 or > 20% vols > 0.5)", None, None)]
    for ft in sorted({0.2, 0.3, 0.5, a.fd_filt}):
        candidates.append(("Filtered FD > %.2g or raw > %.2g; run needs >= %.0f%% retained"
                           % (ft, a.fd_raw, a.min_retained), ft, a.fd_raw))
    comp = []
    for name, ft, rt in candidates:
        kept_runs, vs_by_sub, pct = 0, {}, []
        for sub, d in store.items():
            vs_by_sub[sub] = 0
            for lab, (raw, filt, _) in d.items():
                task = lab.split("_run-")[0]
                if len(raw) < 0.5 * task_median.get(task, 0):
                    continue
                if ft is None:
                    ok = not legacy_fail(raw)
                else:
                    c = censor_mask(raw, filt, ft, rt)
                    c[0] = False
                    pct.append(100 * (1 - c.mean()))
                    ok = pct[-1] >= a.min_retained
                kept_runs += int(ok)
                vs_by_sub[sub] += int(ok and lab.startswith("VS"))
        comp.append({"rule": name, "runs_kept": kept_runs,
                     "runs_total": sum(len(d) for d in store.values()),
                     "participants_kept": sum(v >= a.min_vs for v in vs_by_sub.values()),
                     "participants_total": len(store),
                     "median_pct_volumes_retained": np.median(pct) if pct else np.nan})
    comp = pd.DataFrame(comp)
    comp.round(1).to_csv(str(out / "rule_comparison.csv"), index=False)

    vq = out / "visual_qc_ratings.csv"
    if not vq.exists():
        cols = ["sub", "rater", "date", "T1w_brain_mask", "T1w_to_MNI_registration", "fieldmap_SDC",
                "BOLD_to_T1w_registration", "OFC_vmPFC_signal_dropout", "NAcc_coverage",
                "carpet_plot_artifacts", "T2star_map_plausible", "overall_rating", "notes"]
        pd.DataFrame([dict({c: "" for c in cols}, sub=s) for s in subs])[cols].to_csv(str(vq), index=False)

    if not a.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        plot_heatmap(runs, a, out / "fd_heatmap.png")
        tdir = out / "fd_traces"
        tdir.mkdir(exist_ok=True)
        for sub, d in store.items():
            plot_traces(sub, d, a, tdir)

    pd.set_option("display.width", 200)
    print("\n==== QC summary (proposed rule) ====")
    n_ex = int(parts["participant_excluded"].sum())
    print("Participants: %d screened, %d excluded, %d retained" % (len(parts), n_ex, len(parts) - n_ex))
    print("Runs: %d screened, %d excluded (%d incomplete/aborted)" % (len(runs), int(runs["run_excluded"].sum()), int(runs["incomplete"].sum())))
    print("\n% of volumes censored, by task:")
    print(runs.groupby("task")["pct_censored"].describe()[["count", "mean", "50%", "max"]].round(1))
    print("\nRule comparison (retention only):")
    print(comp.round(1).to_string(index=False))
    if n_ex:
        print("\nExcluded participants:")
        print(parts.loc[parts["participant_excluded"], ["sub", "participant_reason"]].to_string(index=False))
    fl = runs[runs["run_excluded"]]
    if len(fl):
        print("\nExcluded runs:")
        print(fl[["sub", "run_label", "exclusion_reason"]].to_string(index=False))
    odd = parts[(parts["missing_runs"] != "") | (parts["extra_runs"] != "")]
    if len(odd):
        print("\nRun labels that differ from the expected 7 (run numbers are scan order, not conditions):")
        print(odd[["sub", "missing_runs", "extra_runs"]].to_string(index=False))
    print("\nWrote tables, censor files and figures to %s" % out)


if __name__ == "__main__":
    main()
