# Browser Whisper Benchmark

How long does it take to transcribe audio when Whisper runs **inside the browser**
instead of on a server?

This repository publishes measured wall-clock timings for
`onnx-community/whisper-tiny.en` running through
[@huggingface/transformers.js](https://github.com/huggingface/transformers.js)
on WASM, plus a small script that turns those measurements into an estimate for
any audio length.

The numbers exist because nobody publishes this. Model cards quote token
throughput on GPUs; browser deployments run on one CPU core with a 30-second
chunking loop, and the only way to know is to time it.

## Results

| Run | Audio length | Wall clock | Includes model download + init | Output |
|-----|--------------|-----------:|-------------------------------|--------|
| 1 | 11 s | **16.89 s** | yes (first run on a fresh page) | 120 chars |
| 2 | 11 s | **6.49 s** | no | 120 chars |
| 3 | 30 s | **8.98 s** | no | 318 chars |
| 4 | 60 s | **16.06 s** | no | 603 chars |
| 5 | 120 s | **28.58 s** | no | 1160 chars |

Raw data: [`data/whisper-tiny-en-browser-benchmark.csv`](data/whisper-tiny-en-browser-benchmark.csv)

### Derived figures

Every number below is a calculation from the table above, with the formula shown.
None of them is measured separately.

| Figure | Formula | Result |
|--------|---------|--------|
| Model download + init (run 1 only) | `16.89 − 6.49` | **≈ 10.4 s**, one-off, first visit only |
| Processing cost, 120 s file | `28.58 / 120` | 0.24 s CPU per 1 s of audio |
| Processing cost, 60 s file | `16.06 / 60` | 0.27 s CPU per 1 s of audio |
| Processing cost, 30 s file | `8.98 / 30` | 0.30 s CPU per 1 s of audio |
| Least-squares fit over runs 2–5 | see `estimate.py` | `wall = 3.57 + 0.2075 × audio_seconds`, R² = 0.9963 |

The per-second cost falls as the file gets longer, which means there is a fixed
per-run overhead on top of the linear cost — the 11 s file takes 6.49 s
(0.59 s per audio second) rather than the ~0.21 s the fit predicts at 120 s.
Treat the fitted line as an estimate for files of minutes, not seconds.

## Use the script

```
python estimate.py            # show the fit, measured vs fitted, and a table
python estimate.py 600        # estimate a 10-minute recording
python estimate.py --csv path/to/your-own.csv
```

Standard library only. No dependencies, no network, nothing to install.

```
$ python estimate.py 600
Audio length      : 600 s (10.0 min)
Estimated process : 128.0 s (2.1 min)
Real-time factor  : 0.21x  (1.00x = as long as the audio)
First run adds    : ~10.4 s for model download + init
```

## Method

- **Timer start:** the moment the *Transcribe* button is clicked.
- **Timer stop:** the moment the transcript panel becomes visible **and** holds
  non-empty text.
- **What the timer covers:** audio decode + every 30-second chunk pass + rendering.
  It does **not** cover file selection.
- **Runs 2–5** reuse the model already in the page, so they measure processing alone.
- **Run 1** additionally covers the ~40 MB model download, WASM init and encoder warm-up.

### Environment

| | |
|---|---|
| Page | `https://instascript.app/audio-to-text` |
| Model | `onnx-community/whisper-tiny.en`, transformers.js, WASM, quantised (q8) |
| Chunking | audio processed in 30-second passes |
| Browser | Microsoft Edge (Chromium), headless, viewport 1280×900 |
| OS | Windows 11 (build 26200) |
| CPU | Intel Core (Family 6, Model 158) |
| GPU | none used — no WebGPU in the measurement run |

## What this does and does not tell you

- It is **one machine, five runs, one model**. A different CPU shifts the slope;
  a phone shifts it a lot.
- `whisper-tiny.en` is the smallest English-only Whisper model. Larger models are
  slower by roughly their parameter ratio, not by a constant.
- WebGPU, where available, changes the picture entirely. These numbers are the
  WASM baseline — the floor a browser deployment falls back to.
- Memory, not arithmetic, is what stops very long files: audio is buffered and
  processed in 30 s passes.

## Contributing

Timings from other machines are the most useful contribution. Open a PR with a
row added to the CSV and the environment filled in — CPU, browser, OS, whether
WebGPU was used, and whether the run included the model download. A number with
no environment attached is not comparable and will be asked about.

## Origin

The measurements were taken on [InstaScript](https://instascript.app), a
browser-only transcription tool, and published here separately so anyone sizing
an in-browser Whisper deployment can start from a number instead of a guess.

## License

MIT — see [LICENSE](LICENSE). The CSV is released under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
