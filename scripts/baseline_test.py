"""
Baseline Arabic TTS test using unmodified XTTS-v2.

Generates speech from Arabic text using a built-in speaker embedding.
This script produces the baseline outputs for comparison against
improved versions.

Usage:
    conda activate new-arabic-tts
    python scripts/baseline_test.py
"""

import os
import time
import json
import numpy as np
import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- Configuration ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "base")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "original model")
BENCHMARK_DIR = os.path.join(PROJECT_ROOT, "docs", "benchmarks")
SPEAKER_NAME = "Gilberto Mathias"
SAMPLE_RATE = 24000
SENTENCE_PAUSE = 0.35  # seconds

GENERATION_PARAMS = {
    "temperature": 0.3,
    "top_p": 0.7,
    "repetition_penalty": 10.0,
}

TEST_SENTENCES = [
    "الذكاء الاصطناعي يتطور بسرعة كبيرة، ويدخل في كل مجالات الحياة.",
    "الآلات أصبحت قادرة على التعلم، واتخاذ قرارات معقدة بمفردها.",
    "أكثر من 70% من الشركات الكبرى تستخدم الذكاء الاصطناعي اليوم.",
    "هذه التقنية تساعد الأطباء على تشخيص الأمراض بدقة أعلى.",
    "كما تساعد المعلمين على تقديم تعليم مخصص لكل طالب.",
    "لكن كثيراً من الناس يخشون أن تحل الآلات محل الإنسان في العمل.",
    "الخبراء يرون أن الذكاء الاصطناعي سيخلق وظائف جديدة، لم نعرفها بعد.",
    "التحدي الأكبر هو ضمان استخدام هذه التقنية بشكل أخلاقي وعادل.",
    "الدول الكبرى تتسابق على قيادة هذا المجال وتطويره.",
    "مستقبل البشرية سيتشكل بناءً على كيفية تعاملنا مع هذه التقنية.",
]


def measure_audio_stats(wav, sr):
    """Compute basic audio statistics."""
    duration = len(wav) / sr
    rms = np.sqrt(np.mean(wav**2))
    peak = np.max(np.abs(wav))
    # Simple SNR estimate: ratio of RMS to noise floor (bottom 10% frames)
    frame_size = int(sr * 0.025)
    frames = [wav[i : i + frame_size] for i in range(0, len(wav) - frame_size, frame_size)]
    frame_rms = [np.sqrt(np.mean(f**2)) for f in frames]
    frame_rms.sort()
    noise_floor = np.mean(frame_rms[: max(1, len(frame_rms) // 10)])
    snr = 20 * np.log10(rms / max(noise_floor, 1e-10))
    return {
        "duration_s": round(duration, 2),
        "rms": round(float(rms), 6),
        "peak": round(float(peak), 6),
        "snr_db": round(float(snr), 1),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    print("=" * 60)
    print("XTTS-v2 Arabic Baseline Test")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading XTTS-v2 base model...")
    t0 = time.time()
    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR)
    model.cuda()
    model.eval()
    load_time = time.time() - t0
    print(f"    Model loaded in {load_time:.1f}s")

    # Load speaker
    print(f"\n[2/4] Loading speaker: {SPEAKER_NAME}")
    speakers = torch.load(
        os.path.join(MODEL_DIR, "speakers_xtts.pth"), weights_only=False
    )
    speaker_data = speakers[SPEAKER_NAME]
    gpt_cond_latent = speaker_data["gpt_cond_latent"].cuda()
    speaker_embedding = speaker_data["speaker_embedding"].cuda()

    # Generate each sentence
    print(f"\n[3/4] Generating {len(TEST_SENTENCES)} sentences...")
    pause = np.zeros(int(SAMPLE_RATE * SENTENCE_PAUSE))
    all_wav = []
    per_sentence_stats = []

    for i, text in enumerate(TEST_SENTENCES):
        t0 = time.time()
        out = model.inference(
            text=text,
            language="ar",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **GENERATION_PARAMS,
        )
        gen_time = time.time() - t0
        wav = out["wav"]
        stats = measure_audio_stats(wav, SAMPLE_RATE)
        stats["generation_time_s"] = round(gen_time, 2)
        stats["rtf"] = round(gen_time / stats["duration_s"], 3)
        stats["text"] = text
        stats["text_length"] = len(text)
        per_sentence_stats.append(stats)

        all_wav.append(wav)
        if i < len(TEST_SENTENCES) - 1:
            all_wav.append(pause)

        print(f"    [{i+1:2d}/{len(TEST_SENTENCES)}] {stats['duration_s']:5.2f}s "
              f"(RTF={stats['rtf']:.3f}, SNR={stats['snr_db']:.1f}dB) "
              f"| {text[:40]}...")

    # Save combined output
    final_wav = np.concatenate(all_wav)
    output_path = os.path.join(OUTPUT_DIR, "AIOriginal.wav")
    sf.write(output_path, final_wav, SAMPLE_RATE)

    # Overall stats
    total_stats = measure_audio_stats(final_wav, SAMPLE_RATE)
    total_gen_time = sum(s["generation_time_s"] for s in per_sentence_stats)
    total_stats["total_generation_time_s"] = round(total_gen_time, 2)
    total_stats["rtf"] = round(total_gen_time / total_stats["duration_s"], 3)
    total_stats["num_sentences"] = len(TEST_SENTENCES)
    total_stats["speaker"] = SPEAKER_NAME
    total_stats["generation_params"] = GENERATION_PARAMS
    total_stats["sentence_pause_s"] = SENTENCE_PAUSE
    total_stats["model"] = "XTTS-v2 base (unmodified)"

    # Save benchmark
    benchmark = {
        "metadata": {
            "phase": "Phase 1: Baseline",
            "date": time.strftime("%Y-%m-%d"),
            "model": "XTTS-v2 base",
            "speaker": SPEAKER_NAME,
            "params": GENERATION_PARAMS,
        },
        "overall": total_stats,
        "per_sentence": per_sentence_stats,
    }
    benchmark_path = os.path.join(BENCHMARK_DIR, "baseline.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"\n[4/4] Results")
    print(f"    Output:    {output_path}")
    print(f"    Benchmark: {benchmark_path}")
    print(f"    Duration:  {total_stats['duration_s']}s")
    print(f"    Gen Time:  {total_gen_time:.1f}s")
    print(f"    RTF:       {total_stats['rtf']}")
    print(f"    SNR:       {total_stats['snr_db']} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()
