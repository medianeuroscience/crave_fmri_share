#!/usr/bin/env python
"""
run_firstlevel.py - first-level GLMs for the CRAVE study (one GLM per run, all event types
in the same model), using tedana-denoised data in MNI space.

VS (game) runs - regressors built from crave_bids/.../*_task-VS_run-*_events.tsv
  boss_kill              boss death, impulse. If the events file has a chest_dropped column
                         (yes/no), variable-drop runs are split into boss_kill_drop / boss_kill_nodrop
  anticipation           chest opening, FIXED window (default 7.5 s = shortest animation), so
                         anticipation is identical in length and information for every chest
  anim_rest              rest of the opening animation after the window (nuisance)
  reveal_1/_3/_5         moment the items appear (chest_end_reveal), short event, by magnitude
  continuation           continuation-screen blocks, if present in the events file (nuisance)
LB (loot-box video) runs - text, video_lvl1, video_lvl3, video_lvl5 (+ reveal_lvlN if coded)

Nuisance (every run): 24 motion parameters, top 5 aCompCor, fMRIPrep cosine drift terms,
non-steady-state volumes, and one spike regressor per censored volume (from qc_motion.py).
HRF: SPM canonical (no temporal derivative by default - see --hrf). Noise model: AR(1).

Per run it writes (in --out/<space>_fwhm-<N>/sub-XXX/):
  <run>_contrast-<name>_stat-effect.nii.gz    beta / contrast estimate   (for group models)
  <run>_contrast-<name>_stat-variance.nii.gz  its variance               (for mixed-effects)
  <run>_contrast-<name>_stat-z.nii.gz         z map                      (for looking)
  <run>_design.csv / _design.png              the full design matrix
and one row per run in firstlevel_summary.csv: events per regressor, how many survive censoring,
max correlation between task regressors, VIFs, and flags.

Which runs: VS runs with keep == yes in derivatives/run_map.csv; LB runs with an events file.
Runs excluded by qc_motion.py (run_qc.csv run_excluded) are skipped.

Environment: use the tedana env (Python 3.11) and add nilearn once:
  conda activate tedana && pip install nilearn matplotlib
Usage
  python run_firstlevel.py --task VS --subjects sub-048087 --dry-run   # check inputs + events only
  python run_firstlevel.py --task VS --subjects sub-048087             # pilot one participant
  python run_firstlevel.py --task VS --jobs 2                          # everyone (6 mm, whole-brain)
  python run_firstlevel.py --task VS --fwhm 0 --jobs 2                 # unsmoothed, for ROI analyses
  python run_firstlevel.py --task LB --jobs 2
"""
import argparse
import glob
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*null duration.*")
warnings.filterwarnings("ignore", message=".*design matrices are supplied.*")
warnings.filterwarnings("ignore", message=".*Generation of a mask has been requested.*")

SPACE = "space-MNI152NLin2009cAsym_res-2"
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
VS_TASK_REGS = ["anticipation", "reveal_1", "reveal_3", "reveal_5", "chest_1", "chest_3", "chest_5",
                "boss_kill", "boss_kill_drop", "boss_kill_nodrop"]
VS_NUISANCE_EVENTS = ["anim_rest", "continuation"]
LB_TASK_REGS = ["video_lvl1", "video_lvl3", "video_lvl5", "reveal_lvl1", "reveal_lvl3", "reveal_lvl5", "text"]


# ------------------------------------------------------------------ events
def vs_events(ev, ant_win, reveal_dur, chest_model="split"):
    """Turn the VS events file into GLM events. Returns (events DataFrame, notes list)."""
    out, notes, bad = [], [], 0
    variable_drop = str(ev.get("drop_schedule", pd.Series(["Fixed"])).iloc[0]).lower().startswith("var")
    has_drop = "chest_dropped" in ev.columns
    for _, r in ev.iterrows():
        tt = str(r["trial_type"])
        on, du = float(r["onset"]), float(r["duration"]) if r["duration"] == r["duration"] else 0.0
        if tt == "boss_death":
            name = "boss_kill"
            if variable_drop and has_drop:
                name = "boss_kill_drop" if str(r["chest_dropped"]).strip().lower() in ("yes", "1", "true") \
                    else "boss_kill_nodrop"
            out.append((on, 0.0, name))
        elif tt == "lootbox":
            try:
                mag = str(int(float(r.get("magnitude"))))
            except (TypeError, ValueError):
                mag = ""
            if du <= 0 or mag not in ("1", "3", "5"):
                bad += 1
                continue
            if chest_model == "single":
                out.append((on, du + reveal_dur, "chest_" + mag))
            elif chest_model == "block":
                out.append((on, du, "anticipation"))
                out.append((on + du, reveal_dur, "reveal_" + mag))
            else:
                out.append((on, min(ant_win, du), "anticipation"))
                if du > ant_win:
                    out.append((on + ant_win, du - ant_win, "anim_rest"))
                out.append((on + du, reveal_dur, "reveal_" + mag))
        elif tt in ("continuation", "continuation_screen"):
            out.append((on, du, "continuation"))
    if bad:
        notes.append("%d chest(s) without a valid end time or magnitude left out" % bad)
    df = pd.DataFrame(out, columns=["onset", "duration", "trial_type"])
    dup = df.duplicated(["onset", "trial_type"])
    if dup.any():
        notes.append("%d duplicated event(s) removed (same onset and type): check coding" % int(dup.sum()))
        df = df[~dup]
    if variable_drop and not has_drop:
        notes.append("variable-drop run without chest_dropped column: boss kills not split into drop/no drop")
    if not any(t == "continuation" for _, _, t in out):
        notes.append("no continuation screens in events (they stay in the baseline)")
    return df.reset_index(drop=True), notes


def lb_events(ev):
    """Text slides are left out on purpose: they become the baseline. Modelling text as well
    makes the regressors sum to a near-constant (the run is wall-to-wall stimulus) and the
    model unidentifiable (VIF in the hundreds)."""
    keep = ev[ev["trial_type"].isin([r for r in LB_TASK_REGS if r != "text"])][["onset", "duration", "trial_type"]].copy()
    return keep, ["text slides are the baseline: each video estimate = video vs text"]


# ------------------------------------------------------------------ confounds
def load_confounds(conf_path, censor_path, n_vols, use_acompcor=True, minimal=False):
    c = pd.read_csv(str(conf_path), sep="\t")
    cols = []
    for m in MOTION:
        terms = (m,) if minimal else (m, m + "_derivative1", m + "_power2", m + "_derivative1_power2")
        cols += [x for x in terms if x in c]
    if use_acompcor:
        cols += [x for x in ["a_comp_cor_%02d" % i for i in range(2 if minimal else 5)] if x in c]
    cols += [x for x in c.columns if x.startswith("cosine")]
    cols += [x for x in c.columns if x.startswith("non_steady_state_outlier")]
    X = c[cols].fillna(0.0).astype(float)
    censored = np.zeros(n_vols, bool)
    if censor_path is not None and Path(censor_path).exists():
        cen = pd.read_csv(str(censor_path), sep="\t")
        if len(cen) == n_vols:
            censored = np.array(cen["censored"].to_numpy(), dtype=bool, copy=True)
        else:
            raise ValueError("censor file has %d rows, BOLD has %d volumes" % (len(cen), n_vols))
    idx = np.where(censored)[0]
    if len(idx):
        S = np.zeros((n_vols, len(idx)))
        S[idx, np.arange(len(idx))] = 1.0
        X = pd.concat([X.reset_index(drop=True),
                       pd.DataFrame(S, columns=["spike_%04d" % i for i in idx])], axis=1)
    # drop columns that are constant (e.g. a spike duplicated by a non-steady-state column)
    X = X.loc[:, X.std() > 0]
    X = X.T.drop_duplicates().T
    return X, censored


# ------------------------------------------------------------------ diagnostics
def vif(dm, cols):
    out = {}
    A = np.array(dm.to_numpy(float), copy=True)
    for c in cols:
        j = dm.columns.get_loc(c)
        y = A[:, j]
        Xo = np.column_stack([np.ones(len(A)), np.delete(A, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ beta
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else 1.0
        out[c] = float(1 / (1 - r2)) if r2 < 1 else np.inf
    return out


def usable_events(gev, censored, tr, slice_ref, n_vols, lead=4.0, tail=6.0, min_frac=0.5):
    """An event is usable if at least half of the volumes in its response window
    (onset + 4 s  to  offset + 6 s) are uncensored. Works for impulses and blocks alike."""
    counts = {}
    t_vol = (np.arange(n_vols) + slice_ref) * tr
    for t, g in gev.groupby("trial_type"):
        ok = 0
        for on, du in zip(g["onset"].to_numpy(), g["duration"].to_numpy()):
            w = (t_vol >= on + lead) & (t_vol <= on + du + tail)
            if w.any() and (~censored[w]).mean() >= min_frac:
                ok += 1
        counts[t] = (len(g), ok)
    return counts


# ------------------------------------------------------------------ one run
def process(job, a):
    warnings.filterwarnings("ignore", message=".*null duration.*")
    warnings.filterwarnings("ignore", message=".*design matrices are supplied.*")
    warnings.filterwarnings("ignore", message=".*Generation of a mask has been requested.*")
    from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
    import nibabel as nib
    t0 = time.time()
    row = {"sub": job["sub"], "run": job["label"], "status": "ok"}
    try:
        img = nib.load(job["bold"])
        n_vols = img.shape[-1]
        tr = float(img.header.get_zooms()[3]) or a.tr
        if tr <= 0 or abs(tr - a.tr) > 0.05:
            tr = a.tr
        ev = pd.read_csv(job["events"], sep="\t")
        row["condition"] = str(ev["condition"].iloc[0]) if "condition" in ev and len(ev) else ""
        if a.task == "VS":
            gev, notes = vs_events(ev, a.ant_win, a.reveal_dur, a.chest_model)
            task_regs = VS_TASK_REGS
        else:
            gev, notes = lb_events(ev)
            task_regs = LB_TASK_REGS
        gev = gev[gev["onset"] < n_vols * tr]
        minimal = a.task == "LB" and not a.full_nuisance
        conf, censored = load_confounds(job["confounds"], job["censor"], n_vols, not a.no_acompcor, minimal)
        row["n_nuisance"] = conf.shape[1]
        frame_times = (np.arange(n_vols) + a.slice_time_ref) * tr
        dm = make_first_level_design_matrix(frame_times, gev, hrf_model=a.hrf,
                                            drift_model=None, add_regs=conf.to_numpy(),
                                            add_reg_names=list(conf.columns))
        present = [c for c in task_regs if c in dm.columns]
        counts = usable_events(gev, censored, tr, a.slice_time_ref, n_vols)
        for c in task_regs + VS_NUISANCE_EVENTS:
            n_all, n_ok = counts.get(c, (0, 0))
            if n_all:
                row["n_" + c] = n_all
                row["usable_" + c] = n_ok
        row["n_vols"] = n_vols
        row["pct_censored"] = round(100 * censored.mean(), 1)
        if len(present) > 1:
            cm = np.array(dm[present].corr().abs().to_numpy(), dtype=float, copy=True)
            np.fill_diagonal(cm, 0)
            i, j = np.unravel_index(np.argmax(cm), cm.shape)
            row["max_abs_corr"] = round(float(cm[i, j]), 3)
            row["max_corr_pair"] = "%s~%s" % (present[i], present[j])
        for k, vv in vif(dm, present).items():
            row["vif_" + k] = round(vv, 2)
        flags = list(notes)
        min_ev = 1 if a.task == "LB" else a.min_events      # LB has one event of each type by design
        low = [c for c in present if counts.get(c, (0, 0))[1] < min_ev]
        if low:
            flags.append("fewer than %d usable events: %s" % (min_ev, ", ".join(low)))
        high = [c for c in present if row.get("vif_" + c, 0) > a.vif_warn]
        if high:
            flags.append("VIF > %g: %s" % (a.vif_warn, ", ".join(high)))
        row["flags"] = "; ".join(flags)

        out = Path(a.out_dir) / job["sub"]
        out.mkdir(parents=True, exist_ok=True)
        dm.to_csv(str(out / (job["prefix"] + "_design.csv")))
        if not a.no_plots:
            try:
                import matplotlib
                matplotlib.use("Agg")
                from nilearn.plotting import plot_design_matrix
                ax = plot_design_matrix(dm)
                ax.figure.set_size_inches(12, 8)
                ax.figure.savefig(str(out / (job["prefix"] + "_design.png")), dpi=90, bbox_inches="tight")
                matplotlib.pyplot.close(ax.figure)
            except Exception:
                pass
        if a.dry_run:
            row["status"] = "dry-run (design only)"
            return row

        fwhm = a.fwhm if a.fwhm > 0 else None
        model = FirstLevelModel(t_r=tr, slice_time_ref=a.slice_time_ref, mask_img=job["mask"],
                                smoothing_fwhm=fwhm, noise_model="ar1", hrf_model=a.hrf,
                                drift_model=None, minimize_memory=True)
        model.fit(img, design_matrices=[dm])

        contrasts = {c: c for c in present}
        if a.task == "VS":
            rev = [c for c in ("reveal_1", "reveal_3", "reveal_5") if c in present]
            if len(rev) > 1:
                contrasts["reveal_all"] = "(" + "+".join(rev) + ")/%d" % len(rev)
            if "boss_kill_drop" in present and "boss_kill_nodrop" in present:
                contrasts["boss_drop_vs_nodrop"] = "boss_kill_drop-boss_kill_nodrop"
            ch = [c for c in ("chest_1", "chest_3", "chest_5") if c in present]
            if len(ch) > 1:
                contrasts["chest_all"] = "(" + "+".join(ch) + ")/%d" % len(ch)
            if "anticipation" in present and "boss_kill" in present:
                contrasts["anticipation_vs_boss"] = "anticipation-boss_kill"
        else:
            vids = [c for c in ("video_lvl1", "video_lvl3", "video_lvl5") if c in present]
            if len(vids) > 1:
                contrasts["video_all"] = "(" + "+".join(vids) + ")/%d" % len(vids)
        for name, expr in contrasts.items():
            for otype, tag in (("effect_size", "effect"), ("effect_variance", "variance"), ("z_score", "z")):
                m = model.compute_contrast(expr, output_type=otype)
                m.to_filename(str(out / ("%s_contrast-%s_stat-%s.nii.gz" % (job["prefix"], name, tag))))
        row["contrasts"] = ", ".join(contrasts)
    except Exception as e:
        row["status"] = "FAILED: %s: %s" % (type(e).__name__, e)
    row["minutes"] = round((time.time() - t0) / 60, 1)
    return row


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["VS", "LB"], default="VS")
    ap.add_argument("--root", default="/srv/lab/fmri/crave")
    ap.add_argument("--bold", choices=["tedana", "fmriprep"], default="tedana",
                    help="tedana-denoised (primary) or fMRIPrep optimally combined (sensitivity)")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--fwhm", type=float, default=6.0, help="smoothing in mm (0 = none, for ROI analyses)")
    ap.add_argument("--tr", type=float, default=1.761)
    ap.add_argument("--slice-time-ref", type=float, default=0.5,
                    help="fMRIPrep slice-timing reference (fraction of TR); 0.5 is fMRIPrep's default")
    ap.add_argument("--hrf", default="spm", choices=["spm", "spm + derivative"],
                    help="HRF model. Default spm (no temporal derivative): with the chest's back-to-back "
                         "anticipation / animation / reveal events, derivative terms are nearly collinear "
                         "with the reveal regressor (VIF 40-140 in the pilot)")
    ap.add_argument("--chest-model", choices=["split", "block", "single"], default="split",
                    help="split: fixed anticipation window + animation rest + reveal; block: whole opening as "
                         "anticipation + reveal; single: one chest regressor per magnitude (opening + reveal)")
    ap.add_argument("--ant-win", type=float, default=7.5, help="fixed anticipation window (s), split model only")
    ap.add_argument("--reveal-dur", type=float, default=1.0, help="reveal event duration (s)")
    ap.add_argument("--min-events", type=int, default=1, help="usable events a regressor needs in a run (1 = naturalistic design: one event can be all a condition offers)")
    ap.add_argument("--vif-warn", type=float, default=5.0)
    ap.add_argument("--no-acompcor", action="store_true")
    ap.add_argument("--full-nuisance", action="store_true",
                    help="LB only: use the full 24-motion + 5-aCompCor set (default for LB is 6 motion + 2 aCompCor, "
                         "because LB runs have only ~50 volumes)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="runs fitted at once (each needs ~4-8 GB RAM)")
    ap.add_argument("--dry-run", action="store_true", help="build and check designs, fit nothing")
    ap.add_argument("--out", default=None, help="default: ROOT/derivatives/firstlevel")
    a = ap.parse_args()

    root = Path(a.root)
    bids, fp = root / "crave_bids", root / "derivatives" / "fmriprep"
    ted, qcd = root / "derivatives" / "tedana", root / "derivatives" / "qc"
    tag = "%s_bold-%s_fwhm-%g" % (a.task, a.bold, a.fwhm)
    if a.task == "LB":
        tag += "_base-text" + ("_nuis-full" if a.full_nuisance else "_nuis-min")
    if a.task == "VS":
        tag += "_chest-%s_rev-%g" % (a.chest_model, a.reveal_dur) + ("_hrf-deriv" if "derivative" in a.hrf else "")
    a.out_dir = str(Path(a.out or root / "derivatives" / "firstlevel") / tag)

    keep = None
    rm_path = root / "derivatives" / "run_map.csv"
    if a.task == "VS" and rm_path.exists():
        rm = pd.read_csv(str(rm_path), dtype=str)
        keep = set(rm.loc[rm["keep"] == "yes", "bids_prefix"].dropna())
    excluded = set()
    if (qcd / "run_qc.csv").exists():
        q = pd.read_csv(str(qcd / "run_qc.csv"), dtype=str)
        excluded = set(zip(q.loc[q["run_excluded"] == "True", "sub"], q.loc[q["run_excluded"] == "True", "run_label"]))

    jobs, skipped = [], []
    for evf in sorted(glob.glob(str(bids / "sub-*" / "ses-01" / "func" / ("*_task-%s*_events.tsv" % a.task)))):
        prefix = Path(evf).name.replace("_events.tsv", "")
        sub = prefix.split("_")[0]
        if a.subjects and sub not in a.subjects:
            continue
        mrun = re.search(r"_run-(\d+)", prefix)
        label = "%s_run-%02d" % (a.task, int(mrun.group(1))) if mrun else a.task
        func = fp / sub / "ses-01" / "func"
        if a.bold == "tedana":
            bold = ted / sub / "ses-01" / "func" / ("%s_%s_desc-denoised_bold.nii.gz" % (prefix, SPACE))
        else:
            bold = func / ("%s_%s_desc-preproc_bold.nii.gz" % (prefix, SPACE))
        job = {"sub": sub, "label": label, "prefix": prefix, "events": evf, "bold": str(bold),
               "mask": str(func / ("%s_%s_desc-brain_mask.nii.gz" % (prefix, SPACE))),
               "confounds": str(func / ("%s_desc-confounds_timeseries.tsv" % prefix)),
               "censor": str(qcd / "censor" / ("%s_desc-censor_timeseries.tsv" % prefix))}
        why = None
        if keep is not None and prefix not in keep:
            why = "not keep=yes in run_map.csv"
        elif (sub, label) in excluded:
            why = "excluded by motion QC"
        else:
            missing = [k for k in ("bold", "mask", "confounds", "censor") if not Path(job[k]).exists()]
            if missing:
                why = "missing " + ", ".join(missing)
        if why:
            skipped.append((prefix, why))
        else:
            jobs.append(job)

    print("%s runs ready: %d | skipped: %d | output: %s" % (a.task, len(jobs), len(skipped), a.out_dir))
    for p, w in skipped:
        print("  skip %s: %s" % (p, w))
    if not jobs:
        sys.exit("Nothing to run.")
    try:
        import nilearn  # noqa: F401
    except ImportError:
        sys.exit("nilearn is not installed in this environment: conda activate tedana && pip install nilearn matplotlib")

    rows = []
    if a.jobs > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as pool:
            futs = {pool.submit(process, j, a): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                rows.append(r)
                print("[%d/%d] %s %s: %s" % (i, len(jobs), r["sub"], r["run"], r["status"]))
    else:
        for i, j in enumerate(jobs, 1):
            r = process(j, a)
            rows.append(r)
            print("[%d/%d] %s %s: %s" % (i, len(jobs), r["sub"], r["run"], r["status"]))

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    summ = Path(a.out_dir) / "firstlevel_summary.csv"
    df = pd.DataFrame(rows)
    if summ.exists():
        old = pd.read_csv(str(summ))
        old = old[~old.set_index(["sub", "run"]).index.isin(df.set_index(["sub", "run"]).index)]
        df = pd.concat([old, df], ignore_index=True)
    df.sort_values(["sub", "run"]).to_csv(str(summ), index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 80)
    ok = df[df["status"].astype(str).str.startswith(("ok", "dry"))]
    print("\nDone: %d ok, %d failed. Summary: %s" % (len(ok), len(df) - len(ok), summ))
    if "max_abs_corr" in df:
        print("Max |r| between task regressors: median %.2f, max %.2f"
              % (df["max_abs_corr"].median(), df["max_abs_corr"].max()))
    vcols = [c for c in df.columns if c.startswith("vif_")]
    if vcols:
        print("VIF (median / max):  " + "  ".join("%s %.1f/%.1f" % (c[4:], df[c].median(), df[c].max())
                                                for c in vcols if df[c].notna().any()))
    fl = df[df["flags"].fillna("") != ""] if "flags" in df else df.iloc[0:0]
    if len(fl):
        print("\nRuns with flags:")
        print(fl[["sub", "run", "flags"]].to_string(index=False))
    bad = df[~df["status"].astype(str).str.startswith(("ok", "dry"))]
    if len(bad):
        print("\nFailed:")
        print(bad[["sub", "run", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
