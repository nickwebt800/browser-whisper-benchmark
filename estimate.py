#!/usr/bin/env python3
"""Estimate how long in-browser Whisper transcription takes, from measured runs.

The numbers behind this script are real measurements, not benchmarks copied from
somewhere else. They were taken on https://instascript.app/audio-to-text , which
runs `onnx-community/whisper-tiny.en` inside the page via
@huggingface/transformers.js (WASM, q8 quantisation, no WebGPU, no server).

Environment of the measurement run (from benchmark README):
  - Microsoft Edge (Chromium), headless, viewport 1280x900
  - Windows 11 (build 26200), Intel Core (Family 6, Model 158)
  - audio processed in 30-second passes
  - timer starts on Transcribe click, stops when the transcript panel shows text

Usage:
    python estimate.py              # show the fit and a table of estimates
    python estimate.py 300          # estimate one audio length in seconds
    python estimate.py --csv PATH   # use a different measurement file

The script uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "data", "whisper-tiny-en-browser-benchmark.csv")

MODEL_DOWNLOAD_NOTE = (
    "Add ~10.4 s on the very first transcription of a fresh page load: that run "
    "also pays for the ~40 MB model download, WASM init and encoder warm-up. "
    "Measured as run 1 (16.89 s) minus run 2 (6.49 s) on the same 11 s file."
)


def load_rows(path: str) -> list[dict]:
    """Read the measurement CSV. Returns rows with numbers coerced to float."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["audio_seconds"] = float(r["audio_seconds"])
        r["wall_clock_seconds"] = float(r["wall_clock_seconds"])
        r["includes_model_load"] = r["includes_model_load"].strip().lower() == "yes"
    return rows


def steady_rows(rows: list[dict]) -> list[dict]:
    """Runs that measure processing alone (model already loaded in the page)."""
    return [r for r in rows if not r["includes_model_load"]]


def fit_linear(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Least-squares fit y = a + b*x. Returns (intercept, slope, r_squared)."""
    n = len(points)
    if n < 2:
        raise ValueError("need at least two points to fit a line")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
    return intercept, slope, r2


def estimate(intercept: float, slope: float, audio_seconds: float) -> float:
    return intercept + slope * audio_seconds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("seconds", nargs="?", type=float,
                    help="audio length to estimate, in seconds")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="path to the measurement CSV")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print("measurement file not found: %s" % args.csv, file=sys.stderr)
        return 2

    rows = load_rows(args.csv)
    steady = steady_rows(rows)
    intercept, slope, r2 = fit_linear(
        [(r["audio_seconds"], r["wall_clock_seconds"]) for r in steady]
    )

    if args.seconds is None:
        print("Fit on %d measured runs (model already loaded):" % len(steady))
        print("  wall_clock = %.4f + %.4f * audio_seconds   (R^2 = %.4f)"
              % (intercept, slope, r2))
        print("  i.e. %.3f s of processing per 1 s of audio, plus %.2f s fixed\n"
              % (slope, intercept))
        print("Measured vs fitted:")
        print("  %8s %10s %10s %8s" % ("audio s", "measured", "fitted", "diff"))
        for r in steady:
            pred = estimate(intercept, slope, r["audio_seconds"])
            print("  %8.0f %10.2f %10.2f %+8.2f"
                  % (r["audio_seconds"], r["wall_clock_seconds"], pred,
                     pred - r["wall_clock_seconds"]))
        print("\nEstimates (processing only):")
        print("  %8s %10s" % ("audio s", "estimate"))
        for s in (15, 30, 60, 120, 300, 600, 1800, 3600):
            print("  %8d %10.1f" % (s, estimate(intercept, slope, s)))
        print("\n" + MODEL_DOWNLOAD_NOTE)
        print("\nCaveat: this is one fit to five runs on one machine. It is a "
              "starting number, not a promise. Long recordings are split into "
              "30 s passes, so memory, not arithmetic, is what will stop you "
              "first on very long files.")
        return 0

    secs = args.seconds
    if secs <= 0:
        print("audio length must be positive", file=sys.stderr)
        return 2
    est = estimate(intercept, slope, secs)
    print("Audio length      : %.0f s (%.1f min)" % (secs, secs / 60.0))
    print("Estimated process : %.1f s (%.1f min)" % (est, est / 60.0))
    print("Real-time factor  : %.2fx  (1.00x = as long as the audio)" % (est / secs))
    print("First run adds    : ~10.4 s for model download + init")
    print("\nBasis: linear fit to %d measured runs, R^2 = %.4f. "
          "Single machine, see README." % (len(steady), r2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
