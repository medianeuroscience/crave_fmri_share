#!/usr/bin/env python
"""
run_tedana.py - batch multi-echo ICA denoising (tedana) for the CRAVE study,
followed by a warp of the denoised data to MNI space using fMRIPrep's own transforms.

For every run with five echo-wise outputs from fMRIPrep (--me-output-echos):
  1. tedana on the 5 echoes (native BOLD space, already motion- and distortion-corrected)
  2. antsApplyTransforms (from the fMRIPrep container): boldref -> T1w -> MNI152NLin2009cAsym 2 mm
  3. a summary row: components found, accepted, rejected, variance removed

Outputs (default /srv/lab/fmri/crave/derivatives/tedana):
  sub-XXX/ses-01/func/<run>/                     tedana outputs + tedana_report.html
  sub-XXX/ses-01/func/<run>_space-MNI152NLin2009cAsym_res-2_desc-denoised_bold.nii.gz
  logs/<run>.log                                 tedana + ANTs output for that run
  tedana_summary.csv                             one row per run

Finished runs are skipped on re-runs, so it is safe to stop and restart.

Setup (once): tedana needs Python >= 3.8, so use its own environment
  conda create -n tedana python=3.11 -y
  conda activate tedana
  pip install tedana pandas

Usage
  conda activate tedana
  python run_tedana.py --dry-run                         # list what it found; runs nothing
  python run_tedana.py --subjects sub-048087             # pilot one participant first
  python run_tedana.py --jobs 2                          # everyone, 2 runs at a time
Run it inside tmux or screen: the full sample takes many hours.
"""
import argparse
import glob
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

DEFAULT_TE_MS = [14.20, 38.92, 63.64, 88.36, 113.08]
SPACE = "space-MNI152NLin2009cAsym_res-2"


# --------------------------------------------------------------- discovery
def find_runs(fp, bids, subjects, skip, qc_csv, min_vols):
    incomplete = set()
    if qc_csv and Path(qc_csv).exists():
        qc = pd.read_csv(qc_csv)
        if "incomplete" in qc.columns:
            for _, r in qc[qc["incomplete"] == True].iterrows():  # noqa: E712
                incomplete.add((r["sub"], r["run_label"]))

    runs, problems = [], []
    subs = sorted(p.name for p in Path(fp).glob("sub-*") if p.is_dir())
    if subjects:
        subs = [s for s in subs if s in subjects]
    subs = [s for s in subs if s not in skip]
    for sub in subs:
        sdir = Path(fp) / sub
        func = sdir / "ses-01" / "func"
        anat_x = sorted(glob.glob(str(sdir / "**" / "anat" /
                                      "*from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"), recursive=True))
        for e1 in sorted(func.glob("*_echo-1_desc-preproc_bold.nii.gz")):
            if "space-" in e1.name:
                continue
            prefix = e1.name.split("_echo-1_")[0]          # sub-X_ses-01_task-VS_run-01
            task = prefix.split("task-")[1].split("_")[0]
            run = prefix.split("_run-")[1] if "_run-" in prefix else None
            label = "%s_run-%02d" % (task, int(run)) if run else task
            info = {"sub": sub, "prefix": prefix, "run_label": label, "func": str(func)}

            if (sub, label) in incomplete:
                problems.append((prefix, "skipped: incomplete/aborted run in run_qc.csv"))
                continue
            echoes = [func / ("%s_echo-%d_desc-preproc_bold.nii.gz" % (prefix, i)) for i in range(1, 6)]
            if not all(e.exists() for e in echoes):
                problems.append((prefix, "skipped: not all 5 echo files found"))
                continue
            info["echoes"] = [str(e) for e in echoes]

            # echo times from the raw BIDS sidecars (fallback: the confirmed protocol values)
            tes = []
            for i in range(1, 6):
                j = Path(bids) / sub / "ses-01" / "func" / ("%s_echo-%d_bold.json" % (prefix, i))
                try:
                    tes.append(round(json.load(open(str(j)))["EchoTime"] * 1000, 2))
                except Exception:
                    tes = list(DEFAULT_TE_MS)
                    break
            info["tes"] = tes

            # native-space brain mask (tedana makes its own if none is found)
            masks = [m for m in func.glob(prefix + "_*desc-brain_mask.nii.gz") if "space-" not in m.name]
            info["mask"] = str(masks[0]) if masks else None

            xfm = sorted(func.glob(prefix + "_from-boldref_to-T1w_mode-image*_xfm.txt"))
            ref = func / ("%s_%s_boldref.nii.gz" % (prefix, SPACE))
            if not xfm or not anat_x or not ref.exists():
                problems.append((prefix, "tedana only: MNI transforms or reference not found"))
                info["xfm_bold2t1"] = info["xfm_t12mni"] = info["ref"] = None
            else:
                info["xfm_bold2t1"], info["xfm_t12mni"], info["ref"] = str(xfm[0]), anat_x[0], str(ref)

            try:
                import nibabel as nib
                info["n_vols"] = nib.load(info["echoes"][0]).shape[-1]
            except Exception:
                info["n_vols"] = None
            if info["n_vols"] is not None and info["n_vols"] < min_vols:
                problems.append((prefix, "skipped: only %d volumes (< %d)" % (info["n_vols"], min_vols)))
                continue
            runs.append(info)
    return runs, problems


# --------------------------------------------------------------- one run
def process(info, a):
    out = Path(a.out)
    rdir = out / info["sub"] / "ses-01" / "func" / info["prefix"]
    mni = out / info["sub"] / "ses-01" / "func" / ("%s_%s_desc-denoised_bold.nii.gz" % (info["prefix"], SPACE))
    log = out / "logs" / (info["prefix"] + ".log")
    row = {"sub": info["sub"], "run_label": info["run_label"], "n_vols": info["n_vols"],
           "tes_ms": " ".join(str(t) for t in info["tes"]), "mask": "fmriprep" if info["mask"] else "tedana"}
    t0 = time.time()

    def run(cmd, fh):
        fh.write("\n$ " + " ".join(cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode

    with open(str(log), "a") as fh:
        denoised = sorted(rdir.glob("*desc-denoised_bold.nii.gz")) if rdir.exists() else []
        if not denoised:
            if rdir.exists():
                shutil.rmtree(str(rdir))          # clear a half-finished attempt
            rdir.parent.mkdir(parents=True, exist_ok=True)   # tedana only creates the last folder
            cmd = (["tedana", "-d"] + info["echoes"] + ["-e"] + [str(t) for t in info["tes"]] +
                   ["--out-dir", str(rdir), "--prefix", info["prefix"],
                    "--tedpca", a.tedpca, "--n-threads", str(a.threads)])
            if info["mask"]:
                cmd += ["--mask", info["mask"]]
            if run(cmd, fh) != 0:
                row["status"] = "FAILED tedana (see log)"
                return row
            denoised = sorted(rdir.glob("*desc-denoised_bold.nii.gz"))
            if not denoised:
                row["status"] = "FAILED: no denoised output"
                return row

        if info.get("ref") and not mni.exists():
            cmd = ["apptainer", "exec", "--cleanenv", "-B", "%s:%s" % (a.root, a.root), a.sif,
                   "antsApplyTransforms", "-d", "3", "-e", "3",
                   "-i", str(denoised[0]), "-r", info["ref"],
                   "-t", info["xfm_t12mni"], "-t", info["xfm_bold2t1"],
                   "-n", "LanczosWindowedSinc", "--float", "1", "-o", str(mni)]
            if run(cmd, fh) != 0:
                row["status"] = "FAILED warp (see log)"
                return row

    # component summary from tedana's metrics table
    met = sorted(rdir.glob("*desc-tedana_metrics.tsv"))
    if met:
        m = pd.read_csv(str(met[0]), sep="\t")
        cls = m["classification"].astype(str)
        row["n_components"] = len(m)
        row["n_accepted"] = int((cls == "accepted").sum())
        row["n_rejected"] = int((cls == "rejected").sum())
        vcol = [c for c in m.columns if c.lower().replace("_", " ") == "variance explained"]
        if vcol:
            row["pct_variance_rejected"] = round(float(m.loc[cls == "rejected", vcol[0]].sum()), 1)
    row["mni_output"] = "yes" if mni.exists() else "no"
    row["minutes"] = round((time.time() - t0) / 60, 1)
    row["status"] = "ok"
    return row


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/srv/lab/fmri/crave")
    ap.add_argument("--fmriprep", default=None, help="default: ROOT/derivatives/fmriprep")
    ap.add_argument("--bids", default=None, help="default: ROOT/crave_bids")
    ap.add_argument("--out", default=None, help="default: ROOT/derivatives/tedana")
    ap.add_argument("--sif", default=None, help="default: ROOT/containers/fmriprep-25.2.6.sif")
    ap.add_argument("--qc", default=None, help="run_qc.csv to skip aborted runs (default: ROOT/derivatives/qc/run_qc.csv)")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--tedpca", default="aic", help="tedana PCA method (default aic)")
    ap.add_argument("--min-vols", type=int, default=30, help="skip runs shorter than this")
    ap.add_argument("--jobs", type=int, default=2, help="runs processed at once (each needs ~8-12 GB RAM)")
    ap.add_argument("--threads", type=int, default=4, help="threads per tedana run")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    r = Path(a.root)
    a.fmriprep = a.fmriprep or str(r / "derivatives" / "fmriprep")
    a.bids = a.bids or str(r / "crave_bids")
    a.out = a.out or str(r / "derivatives" / "tedana")
    a.sif = a.sif or str(r / "containers" / "fmriprep-25.2.6.sif")
    a.qc = a.qc or str(r / "derivatives" / "qc" / "run_qc.csv")

    runs, problems = find_runs(a.fmriprep, a.bids, a.subjects, a.skip, a.qc, a.min_vols)
    print("Found %d runs ready for tedana across %d participants" % (len(runs), len({x["sub"] for x in runs})))
    for p, why in problems:
        print("  %s: %s" % (p, why))
    short = [x for x in runs if x["n_vols"] and x["n_vols"] < 100]
    if short:
        print("  NOTE: %d runs have < 100 volumes (mostly LB). ICA on so few volumes is less stable;"
              " check their tedana reports, and consider fMRIPrep's optimally combined data for LB." % len(short))

    if a.dry_run:
        ex = runs[0] if runs else None
        if ex:
            print("\nExample (%s):" % ex["prefix"])
            for k in ("echoes", "tes", "mask", "xfm_bold2t1", "xfm_t12mni", "ref", "n_vols"):
                print("  %-12s %s" % (k, ex[k]))
        if shutil.which("tedana") is None:
            print("\nWARNING: 'tedana' is not on your PATH - activate the tedana environment.")
        if not Path(a.sif).exists():
            print("WARNING: container not found at %s" % a.sif)
        return

    if shutil.which("tedana") is None:
        sys.exit("'tedana' not found - run: conda activate tedana")
    (Path(a.out) / "logs").mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        futs = {pool.submit(process, x, a): x for x in runs}
        for i, f in enumerate(as_completed(futs), 1):
            x = futs[f]
            try:
                row = f.result()
            except Exception as e:  # keep going if one run breaks
                row = {"sub": x["sub"], "run_label": x["run_label"], "status": "ERROR: %s" % e}
            rows.append(row)
            print("[%d/%d] %s %s: %s" % (i, len(runs), row["sub"], row["run_label"], row["status"]))

    summ = Path(a.out) / "tedana_summary.csv"
    df = pd.DataFrame(rows)
    if summ.exists():  # keep rows from earlier sessions for runs not redone now
        old = pd.read_csv(str(summ))
        keep = ~old.set_index(["sub", "run_label"]).index.isin(df.set_index(["sub", "run_label"]).index)
        df = pd.concat([old[keep], df], ignore_index=True)
    df.sort_values(["sub", "run_label"]).to_csv(str(summ), index=False)
    bad = df[df["status"] != "ok"]
    print("\nDone: %d ok, %d failed. Summary: %s" % ((df["status"] == "ok").sum(), len(bad), summ))
    if len(bad):
        print(bad[["sub", "run_label", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
