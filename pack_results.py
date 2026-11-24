import argparse
import sys
import zipfile
from datetime import datetime
from pathlib import Path

FINAL_FOLDERS = (
    "experiment_results_final",
    "three_task_final_results",
    "experiment_results_smolvlm",
    "experiment_results_gemma",
    "three_task_return_a_results",
    "three_task_return_b_results",
)
RESULT_FOLDERS = (
    "experiment_results",
    "three_task_box_memory_results",
    "three_task_fixed_return_results",
    "experiment_results_hybrid",
    "smoke_test_results",
    "experiment_results_final",
    "three_task_final_results",
    "final_check_results",
    "experiment_results_smolvlm",
    "experiment_results_gemma",
    "three_task_return_a_results",
    "three_task_return_b_results",
)
KEEP_SUFFIXES = {".json", ".csv", ".txt"}
SKIP_FOLDERS = {"captures", "manual_review_images", "__pycache__"}


def newest_runs(folders=FINAL_FOLDERS):
    runs = []
    for folder in folders:
        base = Path(folder)
        if not base.is_dir():
            continue
        candidates = [p for p in base.iterdir() if p.is_dir()]
        if candidates:
            runs.append(max(candidates, key=lambda p: p.stat().st_mtime))
    return runs


def files_to_keep(run):
    for path in sorted(run.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run)
        if any(part in SKIP_FOLDERS for part in relative.parts[:-1]):
            continue
        if path.suffix.lower() in KEEP_SUFFIXES:
            yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Zip run results without images.")
    parser.add_argument("runs", nargs="*", help="Run folders (default: newest run of each final results folder).")
    parser.add_argument("--all", action="store_true", help="Use the newest run of every results folder, including the paper-version ones.")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra files to include, e.g. replay_results.csv.")
    parser.add_argument("--out", help="Output zip name (default: results_<timestamp>.zip).")
    args = parser.parse_args(argv)

    if args.runs:
        runs = [Path(r) for r in args.runs]
    else:
        runs = newest_runs(RESULT_FOLDERS if args.all else FINAL_FOLDERS)
    missing = [r for r in runs if not r.is_dir()]
    if missing:
        parser.error("not a folder: " + ", ".join(str(m) for m in missing))
    if not runs and not args.extra:
        parser.error("no run folders found; pass them explicitly")

    out = Path(args.out or f"results_{datetime.now():%Y%m%d_%H%M%S}.zip")
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for run in runs:
            for path in files_to_keep(run):
                archive.write(path, Path(run.parent.name) / run.name / path.relative_to(run))
                count += 1
            print(f"added {run}")
        for extra in args.extra:
            path = Path(extra)
            if path.is_file():
                archive.write(path, path.name)
                count += 1
                print(f"added {path}")
            else:
                print(f"skipped {path}: not found")

    size_mb = out.stat().st_size / 1_000_000
    print(f"\nWrote {out} ({count} files, {size_mb:.1f} MB)")
    if size_mb > 30:
        print("Still large: the per-step history JSON is the bulk; upload it anyway "
              "or split the runs into separate zips.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
