"""
Phase 7: Evaluation & Comparison

Generates quantitative metrics and comparison charts for all three
model stages: baseline, improved pipeline, and fine-tuned.

Metrics:
  - Audio duration and pacing analysis
  - Signal-to-noise ratio (SNR)
  - Real-time factor (RTF)
  - Spectral analysis (energy distribution)
  - Waveform comparison charts

Output:
  docs/images/comparison_waveforms.png
  docs/images/comparison_spectrogram.png
  docs/images/comparison_duration.png
  docs/images/training_loss.png
  docs/benchmarks/evaluation.json

Usage:
    conda activate new-arabic-tts
    python scripts/evaluate.py
"""

import os
import json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

# Output files from each phase
FILES = {
    "Phase 1: Baseline": PROJECT_ROOT / "outputs" / "base_model_test.wav",
    "Phase 6: Fine-tuned": PROJECT_ROOT / "outputs" / "finetuned_model_test.wav",
}

COLORS = {
    "Phase 1: Baseline": "#e74c3c",
    "Phase 6: Fine-tuned": "#2ecc71",
}


def measure_audio(wav, sr):
    """Compute audio metrics."""
    duration = len(wav) / sr
    rms = np.sqrt(np.mean(wav ** 2))
    peak = np.max(np.abs(wav))

    # SNR
    frame_size = int(sr * 0.025)
    hop = frame_size // 2
    frames = [wav[i:i + frame_size] for i in range(0, len(wav) - frame_size, hop)]
    rms_vals = sorted([np.sqrt(np.mean(f ** 2)) for f in frames])
    n = len(rms_vals)
    noise = np.mean(rms_vals[:max(1, n // 10)])
    signal = np.mean(rms_vals[n // 2:])
    snr = 20 * np.log10(signal / max(noise, 1e-10))

    # Silence ratio
    silence = np.mean(np.abs(wav) < 0.01)

    # Speaking rate (non-silent frames)
    speaking_frames = sum(1 for f in frames if np.sqrt(np.mean(f ** 2)) > 0.01)
    speaking_duration = speaking_frames * (hop / sr)

    return {
        "duration_s": round(duration, 2),
        "rms": round(float(rms), 6),
        "peak": round(float(peak), 6),
        "snr_db": round(float(snr), 1),
        "silence_ratio": round(float(silence), 3),
        "speaking_duration_s": round(speaking_duration, 2),
    }


def plot_waveforms(audio_data, sr_data):
    """Side-by-side waveform comparison."""
    n_plots = len(audio_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (name, wav) in zip(axes, audio_data.items()):
        sr = sr_data[name]
        time_axis = np.arange(len(wav)) / sr
        color = COLORS.get(name, "#3498db")

        ax.plot(time_axis, wav, color=color, linewidth=0.3, alpha=0.8)
        ax.fill_between(time_axis, wav, alpha=0.15, color=color)
        ax.set_ylabel("Amplitude")
        ax.set_title(name, fontsize=11, fontweight="bold", color=color)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, max(len(w) / sr_data[n] for n, w in audio_data.items()))
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0, color="gray", linewidth=0.5)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Waveform Comparison — Base vs Fine-tuned", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparison_waveforms.png", dpi=150)
    plt.close(fig)
    print("  Saved comparison_waveforms.png")


def plot_spectrograms(audio_data, sr_data):
    """Spectrogram comparison."""
    n_plots = len(audio_data)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, wav) in zip(axes, audio_data.items()):
        sr = sr_data[name]
        ax.specgram(wav, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 8000)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("Spectrogram Comparison — Base vs Fine-tuned", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparison_spectrogram.png", dpi=150)
    plt.close(fig)
    print("  Saved comparison_spectrogram.png")


def plot_metrics_comparison(metrics):
    """Bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    names = list(metrics.keys())
    short_names = [n.split(": ")[-1] for n in names]
    colors = [COLORS.get(n, "#3498db") for n in names]

    # Duration
    vals = [metrics[n]["duration_s"] for n in names]
    axes[0].bar(short_names, vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Duration (seconds)")
    axes[0].set_ylabel("Seconds")
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.2, f"{v}s", ha="center", fontsize=10, fontweight="bold")

    # SNR
    vals = [metrics[n]["snr_db"] for n in names]
    axes[1].bar(short_names, vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Signal-to-Noise Ratio")
    axes[1].set_ylabel("dB")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.5, f"{v}", ha="center", fontsize=10, fontweight="bold")

    # Speaking duration (non-silence)
    vals = [metrics[n]["speaking_duration_s"] for n in names]
    axes[2].bar(short_names, vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[2].set_title("Speaking Duration")
    axes[2].set_ylabel("Seconds")
    for i, v in enumerate(vals):
        axes[2].text(i, v + 0.2, f"{v}s", ha="center", fontsize=10, fontweight="bold")

    # Silence ratio
    vals = [metrics[n]["silence_ratio"] * 100 for n in names]
    axes[3].bar(short_names, vals, color=colors, edgecolor="white", linewidth=1.5)
    axes[3].set_title("Silence Ratio")
    axes[3].set_ylabel("%")
    for i, v in enumerate(vals):
        axes[3].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Audio Metrics Comparison — Base vs Fine-tuned",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "comparison_metrics.png", dpi=150)
    plt.close(fig)
    print("  Saved comparison_metrics.png")


def plot_training_loss():
    """Plot training loss curve from trainer log."""
    # Find trainer log
    ft_dir = PROJECT_ROOT / "models" / "finetuned" / "run" / "training"
    log_files = list(ft_dir.rglob("trainer_0_log.txt"))
    if not log_files:
        print("  No trainer log found, skipping loss plot")
        return

    log_path = log_files[0]
    steps, losses = [], []
    current_step = 0

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            # Parse step from: --> TIME: ... -- STEP: 50/1082 -- GLOBAL_STEP: 50
            if "GLOBAL_STEP:" in line:
                try:
                    current_step = int(line.split("GLOBAL_STEP:")[1].strip().rstrip("\x1b[0m"))
                except (ValueError, IndexError):
                    pass
            # Parse loss from:  | > loss: 2.543 (2.543)
            if "| > loss:" in line and "loss_text" not in line and "loss_mel" not in line:
                try:
                    val = float(line.split("loss:")[1].strip().split()[0])
                    if current_step > 0:
                        steps.append(current_step)
                        losses.append(val)
                except (ValueError, IndexError):
                    pass

    if not steps:
        print("  No loss data in log, skipping loss plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, color="#2ecc71", linewidth=1.5, alpha=0.8)

    # Smoothed line
    if len(losses) > 20:
        window = min(50, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], smoothed, color="#27ae60", linewidth=2.5, label="Smoothed")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve — XTTS-v2 Fine-tuning on Egyptian Arabic", fontsize=13)
    ax.grid(True, alpha=0.2)
    if len(losses) > 20:
        ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "training_loss.png", dpi=150)
    plt.close(fig)
    print("  Saved training_loss.png")


def main():
    print("=" * 70)
    print("  Phase 7: Evaluation & Comparison")
    print("=" * 70)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # Load audio files
    print("\n[1/5] Loading audio files...")
    audio_data = {}
    sr_data = {}
    metrics = {}

    for name, path in FILES.items():
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        wav, sr = sf.read(str(path))
        audio_data[name] = wav
        sr_data[name] = sr
        m = measure_audio(wav, sr)
        metrics[name] = m
        print(f"  {name}: {m['duration_s']}s, SNR={m['snr_db']}dB, silence={m['silence_ratio']:.1%}")

    # Generate charts
    print("\n[2/5] Generating waveform comparison...")
    plot_waveforms(audio_data, sr_data)

    print("\n[3/5] Generating spectrogram comparison...")
    plot_spectrograms(audio_data, sr_data)

    print("\n[4/5] Generating metrics comparison...")
    plot_metrics_comparison(metrics)

    print("\n[5/5] Generating training loss curve...")
    plot_training_loss()

    # Save evaluation report
    report = {
        "date": __import__("time").strftime("%Y-%m-%d"),
        "files": {name: str(path) for name, path in FILES.items()},
        "metrics": metrics,
    }
    report_path = BENCHMARKS_DIR / "evaluation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print("  Evaluation Complete!")
    print(f"{'='*70}")
    print(f"\n  Charts saved to: {IMAGES_DIR}/")
    print(f"    - comparison_waveforms.png")
    print(f"    - comparison_spectrogram.png")
    print(f"    - comparison_metrics.png")
    print(f"    - training_loss.png")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
