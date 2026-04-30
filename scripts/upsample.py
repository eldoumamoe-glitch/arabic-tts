"""
Phase 8: Studio-Quality Audio Upsampling

Uses AudioSR to upsample XTTS-v2 output from 24kHz to 48kHz,
reconstructing high-frequency detail that the base model cannot produce.

The XTTS-v2 model outputs 24kHz audio. Changing the internal sample rate
would require retraining the entire model (~370M params) from scratch.
Instead, we use a neural upsampler as a post-processing step — the same
approach used in professional TTS production pipelines.

Usage:
    conda activate new-arabic-tts
    python scripts/upsample.py --input outputs/finetuned_model_test.wav

Output:
    Upsampled WAV saved alongside the input with _48kHz suffix.

Usage for all outputs:
    python scripts/upsample.py --all
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import soundfile as sf
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def upsample_audiosr(input_path, output_path):
    """Upsample using AudioSR neural upsampler."""
    import audiosr

    print(f"  Loading AudioSR model...")
    model = audiosr.build_model(model_name="speech", device="cuda")

    print(f"  Upsampling: {input_path}")
    t0 = time.time()
    waveform = audiosr.super_resolution(
        model,
        str(input_path),
        seed=42,
        guidance_scale=3.5,
        ddim_steps=50,
    )
    elapsed = time.time() - t0

    # AudioSR returns [batch, channels, samples] at 48kHz
    if hasattr(waveform, 'cpu'):
        wav = waveform[0, 0].cpu().numpy()
    else:
        wav = waveform[0, 0]
    target_sr = 48000

    sf.write(str(output_path), wav, target_sr, subtype="PCM_24")
    duration = len(wav) / target_sr
    print(f"  Saved: {output_path}")
    print(f"  Duration: {duration:.2f}s, Sample rate: {target_sr}Hz, Bit depth: 24-bit")
    print(f"  Processing time: {elapsed:.1f}s")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_sr": 24000,
        "output_sr": target_sr,
        "bit_depth": 24,
        "duration_s": round(duration, 2),
        "processing_time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Audio Upsampling (24kHz → 48kHz)")
    parser.add_argument("--input", type=str, help="Input WAV file to upsample")
    parser.add_argument("--all", action="store_true", help="Upsample all official output files")
    args = parser.parse_args()

    if not args.input and not args.all:
        parser.error("Provide --input <file> or --all")

    print("=" * 70)
    print("  Phase 8: Studio-Quality Audio Upsampling")
    print("  24kHz / 16-bit → 48kHz / 24-bit")
    print("=" * 70)

    results = []

    if args.all:
        files = [
            PROJECT_ROOT / "outputs" / "base_model_test.wav",
            PROJECT_ROOT / "outputs" / "finetuned_model_test.wav",
        ]
    else:
        files = [Path(args.input)]

    for input_path in files:
        if not input_path.exists():
            print(f"\n  WARNING: {input_path} not found, skipping")
            continue

        output_path = input_path.with_stem(input_path.stem + "_48kHz")
        print(f"\n  [{files.index(input_path)+1}/{len(files)}] {input_path.name}")
        result = upsample_audiosr(input_path, output_path)
        results.append(result)

    # Save report
    benchmarks_dir = PROJECT_ROOT / "docs" / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    report_path = benchmarks_dir / "upsampling.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"date": time.strftime("%Y-%m-%d"), "results": results}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  Upsampling Complete!")
    print(f"  Report: {report_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
