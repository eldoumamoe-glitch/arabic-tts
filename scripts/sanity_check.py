"""
Data sanity check pipeline for Arabic TTS training data.

Clean data is crucial for TTS fine-tuning. Unlike ASR (speech recognition),
where the model can tolerate some noise and still learn to transcribe,
TTS models learn to *reproduce* exactly what they hear. If the training
data contains:
  - Misaligned text (text doesn't match audio) → model learns wrong pronunciations
  - Noisy audio → model learns to generate noise
  - Silence or dead air → model learns to produce silence mid-speech
  - Corrupted text → model learns to mispronounce words

A single bad clip won't ruin the model, but hundreds of bad clips in a
5,000-clip dataset will noticeably degrade output quality. This script
applies three layers of automated verification to catch problems before
they reach the training pipeline.

Three-layer verification:
  Layer 1 — Text quality: character validation, length ratios, pattern detection
  Layer 2 — Audio quality: SNR, silence ratio, clipping detection
  Layer 3 — Alignment verification: Whisper cross-check (independent transcript)

Usage:
    conda activate new-arabic-tts
    python scripts/sanity_check.py

Output:
    data/egyptian/metadata_train.csv        (cleaned, bad clips removed)
    data/egyptian/metadata_eval.csv         (cleaned, bad clips removed)
    data/egyptian/rejected/                 (moved bad clips for review)
    docs/benchmarks/sanity_check.json       (full report)
"""

import os
import re
import sys
import csv
import json
import time
import shutil
import numpy as np
import soundfile as sf
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "egyptian"
WAVS_DIR = DATA_DIR / "wavs"
REJECTED_DIR = DATA_DIR / "rejected"
BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

# --- Layer 1: Text Quality Thresholds ---
MIN_ARABIC_RATIO = 0.60        # At least 60% Arabic characters
MAX_LATIN_CHARS = 3            # Allow max 3 Latin chars (typos)
MAX_REPEATED_WORDS = 3         # Flag if same word appears 3+ times in a row
MIN_CHARS_PER_SEC = 3.0        # Minimum characters per second (too few = wrong text)
MAX_CHARS_PER_SEC = 25.0       # Maximum characters per second (too many = wrong text)

# --- Layer 2: Audio Quality Thresholds ---
MIN_SNR_DB = 10.0              # Minimum signal-to-noise ratio
MAX_SILENCE_RATIO = 0.50       # Max 50% silence
CLIPPING_THRESHOLD = 0.99      # Samples above this are clipped
MAX_CLIPPING_RATIO = 0.01      # Max 1% of samples can be clipped

# --- Layer 3: Alignment Thresholds ---
MIN_SIMILARITY = 0.40          # Minimum text similarity between original and Whisper


def is_arabic(char):
    """Check if character is Arabic."""
    return '\u0600' <= char <= '\u06FF' or '\u0750' <= char <= '\u077F' or '\uFB50' <= char <= '\uFDFF' or '\uFE70' <= char <= '\uFEFF'


def arabic_ratio(text):
    """Fraction of characters that are Arabic."""
    text_no_space = text.replace(" ", "")
    if not text_no_space:
        return 0
    return sum(1 for c in text_no_space if is_arabic(c)) / len(text_no_space)


def count_latin(text):
    """Count Latin alphabet characters."""
    return sum(1 for c in text if c.isascii() and c.isalpha())


def has_repeated_words(text, threshold=MAX_REPEATED_WORDS):
    """Detect stuttered/repeated words."""
    words = text.split()
    for i in range(len(words) - threshold + 1):
        if len(set(words[i:i+threshold])) == 1:
            return True
    return False


def measure_snr(wav, sr):
    """Estimate SNR from frame-level RMS."""
    frame_size = int(sr * 0.025)
    hop = frame_size // 2
    frames = [wav[i:i+frame_size] for i in range(0, len(wav) - frame_size, hop)]
    if not frames:
        return 0
    rms_values = [np.sqrt(np.mean(f**2)) for f in frames]
    rms_values.sort()
    n = len(rms_values)
    noise_floor = np.mean(rms_values[:max(1, n // 10)])
    signal_level = np.mean(rms_values[n // 2:])
    if noise_floor < 1e-10:
        return 100
    return 20 * np.log10(signal_level / noise_floor)


def silence_ratio(wav, sr, threshold=0.01):
    """Fraction of audio that is silence."""
    return np.mean(np.abs(wav) < threshold)


def clipping_ratio(wav):
    """Fraction of samples that are clipped."""
    return np.mean(np.abs(wav) >= CLIPPING_THRESHOLD)


def normalize_arabic(text):
    """Normalize Arabic text for comparison."""
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    # Normalize alef variants
    text = re.sub(r'[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = text.replace('ة', 'ه')
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_similarity(text1, text2):
    """Compare two Arabic texts after normalization."""
    t1 = normalize_arabic(text1)
    t2 = normalize_arabic(text2)
    return SequenceMatcher(None, t1, t2).ratio()


def load_metadata(csv_path):
    """Load pipe-delimited metadata CSV."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def save_metadata(rows, csv_path):
    """Save pipe-delimited metadata CSV."""
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\n")
        for row in rows:
            f.write(f"{row['audio_file']}|{row['text']}|{row['speaker_name']}\n")


def main():
    print("=" * 70)
    print("  Data Sanity Check Pipeline")
    print("  Three-layer verification for TTS training data")
    print("=" * 70)

    t_start = time.time()
    REJECTED_DIR.mkdir(parents=True, exist_ok=True)

    # Load both splits
    train_rows = load_metadata(DATA_DIR / "metadata_train.csv")
    eval_rows = load_metadata(DATA_DIR / "metadata_eval.csv")
    all_rows = train_rows + eval_rows
    total_clips = len(all_rows)
    print(f"\nLoaded {len(train_rows)} train + {len(eval_rows)} eval = {total_clips} clips")

    rejected = {}  # clip -> reason

    # ===== LAYER 1: TEXT QUALITY =====
    print(f"\n{'─'*50}")
    print("  Layer 1: Text Quality Check")
    print(f"{'─'*50}")

    l1_issues = {"low_arabic_ratio": 0, "too_many_latin": 0,
                 "repeated_words": 0, "bad_char_rate": 0}

    for row in all_rows:
        clip = row["audio_file"]
        text = row["text"]

        if clip in rejected:
            continue

        # Arabic ratio
        ar = arabic_ratio(text)
        if ar < MIN_ARABIC_RATIO:
            rejected[clip] = f"Layer 1: Low Arabic ratio ({ar:.2f})"
            l1_issues["low_arabic_ratio"] += 1
            continue

        # Latin characters
        latin = count_latin(text)
        if latin > MAX_LATIN_CHARS:
            rejected[clip] = f"Layer 1: Too many Latin chars ({latin})"
            l1_issues["too_many_latin"] += 1
            continue

        # Repeated words
        if has_repeated_words(text):
            rejected[clip] = "Layer 1: Repeated words detected"
            l1_issues["repeated_words"] += 1
            continue

        # Characters per second (need audio duration)
        wav_path = DATA_DIR / clip
        if wav_path.exists():
            info = sf.info(str(wav_path))
            duration = info.duration
            chars_per_sec = len(text) / max(duration, 0.1)
            if chars_per_sec < MIN_CHARS_PER_SEC or chars_per_sec > MAX_CHARS_PER_SEC:
                rejected[clip] = f"Layer 1: Abnormal char rate ({chars_per_sec:.1f} chars/s)"
                l1_issues["bad_char_rate"] += 1

    l1_total = sum(l1_issues.values())
    print(f"  Low Arabic ratio:    {l1_issues['low_arabic_ratio']}")
    print(f"  Too many Latin:      {l1_issues['too_many_latin']}")
    print(f"  Repeated words:      {l1_issues['repeated_words']}")
    print(f"  Abnormal char rate:  {l1_issues['bad_char_rate']}")
    print(f"  Total rejected:      {l1_total}")

    # ===== LAYER 2: AUDIO QUALITY =====
    print(f"\n{'─'*50}")
    print("  Layer 2: Audio Quality Check")
    print(f"{'─'*50}")

    l2_issues = {"low_snr": 0, "high_silence": 0, "clipping": 0, "missing_file": 0}

    for i, row in enumerate(all_rows):
        clip = row["audio_file"]
        if clip in rejected:
            continue

        wav_path = DATA_DIR / clip
        if not wav_path.exists():
            rejected[clip] = "Layer 2: File missing"
            l2_issues["missing_file"] += 1
            continue

        wav, sr = sf.read(str(wav_path))

        # SNR
        snr = measure_snr(wav, sr)
        if snr < MIN_SNR_DB:
            rejected[clip] = f"Layer 2: Low SNR ({snr:.1f} dB)"
            l2_issues["low_snr"] += 1
            continue

        # Silence ratio
        sil = silence_ratio(wav, sr)
        if sil > MAX_SILENCE_RATIO:
            rejected[clip] = f"Layer 2: Too much silence ({sil:.0%})"
            l2_issues["high_silence"] += 1
            continue

        # Clipping
        clip_r = clipping_ratio(wav)
        if clip_r > MAX_CLIPPING_RATIO:
            rejected[clip] = f"Layer 2: Audio clipping ({clip_r:.1%})"
            l2_issues["clipping"] += 1

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1:,}/{total_clips}] checked...")

    l2_total = sum(l2_issues.values())
    print(f"  Low SNR:             {l2_issues['low_snr']}")
    print(f"  Too much silence:    {l2_issues['high_silence']}")
    print(f"  Audio clipping:      {l2_issues['clipping']}")
    print(f"  Missing files:       {l2_issues['missing_file']}")
    print(f"  Total rejected:      {l2_total}")

    # ===== LAYER 3: ALIGNMENT VERIFICATION (WHISPER) =====
    print(f"\n{'─'*50}")
    print("  Layer 3: Alignment Verification (Whisper)")
    print(f"{'─'*50}")
    print("  Loading Whisper model (this may take a minute)...")

    import whisper
    whisper_model = whisper.load_model("large-v3", device="cuda")

    l3_issues = {"low_similarity": 0}
    l3_checked = 0
    t0 = time.time()

    for i, row in enumerate(all_rows):
        clip = row["audio_file"]
        if clip in rejected:
            continue

        wav_path = str(DATA_DIR / clip)
        original_text = row["text"]

        try:
            result = whisper_model.transcribe(wav_path, language="ar", fp16=True)
            whisper_text = result["text"].strip()

            sim = text_similarity(original_text, whisper_text)
            if sim < MIN_SIMILARITY:
                rejected[clip] = f"Layer 3: Low alignment ({sim:.2f}) | Original: {original_text[:50]} | Whisper: {whisper_text[:50]}"
                l3_issues["low_similarity"] += 1
        except Exception as e:
            rejected[clip] = f"Layer 3: Whisper error ({str(e)[:50]})"
            l3_issues["low_similarity"] += 1

        l3_checked += 1
        if (i + 1) % 500 == 0:
            rate = l3_checked / (time.time() - t0)
            remaining = sum(1 for r in all_rows[i+1:] if r["audio_file"] not in rejected)
            eta = remaining / max(rate, 1) / 60
            print(f"  [{i+1:,}/{total_clips}] {l3_checked} verified, "
                  f"{l3_issues['low_similarity']} misaligned, "
                  f"{rate:.0f} clips/s, ETA {eta:.0f}min")

    l3_total = sum(l3_issues.values())
    print(f"  Low alignment:       {l3_issues['low_similarity']}")
    print(f"  Total rejected:      {l3_total}")

    # ===== APPLY REJECTIONS =====
    print(f"\n{'─'*50}")
    print("  Applying rejections")
    print(f"{'─'*50}")

    # Move rejected WAVs
    moved = 0
    for clip, reason in rejected.items():
        src = DATA_DIR / clip
        dst = REJECTED_DIR / Path(clip).name
        if src.exists():
            shutil.move(str(src), str(dst))
            moved += 1

    # Filter metadata
    clean_train = [r for r in train_rows if r["audio_file"] not in rejected]
    clean_eval = [r for r in eval_rows if r["audio_file"] not in rejected]

    save_metadata(clean_train, DATA_DIR / "metadata_train.csv")
    save_metadata(clean_eval, DATA_DIR / "metadata_eval.csv")

    # ===== SUMMARY =====
    total_rejected = len(rejected)
    total_kept = total_clips - total_rejected

    print(f"\n{'='*70}")
    print(f"  Sanity Check Complete!")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\n  Results:")
    print(f"    Original clips:     {total_clips:,}")
    print(f"    Rejected:           {total_rejected:,} ({total_rejected/total_clips:.1%})")
    print(f"      Layer 1 (text):   {l1_total:,}")
    print(f"      Layer 2 (audio):  {l2_total:,}")
    print(f"      Layer 3 (align):  {l3_total:,}")
    print(f"    Kept:               {total_kept:,} ({total_kept/total_clips:.1%})")
    print(f"      Train:            {len(clean_train):,}")
    print(f"      Eval:             {len(clean_eval):,}")
    print(f"\n  Files:")
    print(f"    Clean train:  {DATA_DIR / 'metadata_train.csv'}")
    print(f"    Clean eval:   {DATA_DIR / 'metadata_eval.csv'}")
    print(f"    Rejected WAVs: {REJECTED_DIR}/ ({moved} files)")

    # Save review file for Layer 3 rejections (compare original vs Whisper)
    review_path = DATA_DIR / "rejected" / "alignment_review.csv"
    l3_review_count = 0
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("audio_file|original_text|whisper_text|similarity|status\n")
        for clip, reason in sorted(rejected.items()):
            if reason.startswith("Layer 3: Low alignment"):
                # Parse the reason string to extract texts and similarity
                parts = reason.split(" | ")
                sim_str = parts[0].split("(")[1].rstrip(")")
                orig = parts[1].replace("Original: ", "") if len(parts) > 1 else ""
                whis = parts[2].replace("Whisper: ", "") if len(parts) > 2 else ""
                f.write(f"{clip}|{orig}|{whis}|{sim_str}|REJECTED\n")
                l3_review_count += 1
    print(f"\n  Review file for Layer 3 rejections ({l3_review_count} clips):")
    print(f"    {review_path}")
    print(f"    Open this file to compare original text vs Whisper text")
    print(f"    and decide if the rejection was correct.")

    # Save report
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "date": time.strftime("%Y-%m-%d"),
        "total_clips": total_clips,
        "total_rejected": total_rejected,
        "total_kept": total_kept,
        "layer_1_text": {
            "rejected": l1_total,
            "details": l1_issues,
            "thresholds": {
                "min_arabic_ratio": MIN_ARABIC_RATIO,
                "max_latin_chars": MAX_LATIN_CHARS,
                "max_repeated_words": MAX_REPEATED_WORDS,
                "min_chars_per_sec": MIN_CHARS_PER_SEC,
                "max_chars_per_sec": MAX_CHARS_PER_SEC,
            }
        },
        "layer_2_audio": {
            "rejected": l2_total,
            "details": l2_issues,
            "thresholds": {
                "min_snr_db": MIN_SNR_DB,
                "max_silence_ratio": MAX_SILENCE_RATIO,
                "max_clipping_ratio": MAX_CLIPPING_RATIO,
            }
        },
        "layer_3_alignment": {
            "rejected": l3_total,
            "details": l3_issues,
            "whisper_model": "large-v3 (OpenAI)",
            "min_similarity": MIN_SIMILARITY,
            "note": "Original metadata text is NEVER modified. Whisper is only used to verify alignment. Review alignment_review.csv to compare.",
        },
        "output": {
            "train_clips": len(clean_train),
            "eval_clips": len(clean_eval),
        },
        "rejected_clips": {clip: reason for clip, reason in sorted(rejected.items())},
        "pipeline_time_min": round((time.time() - t_start) / 60, 1),
    }
    report_path = BENCHMARKS_DIR / "sanity_check.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"    Report:       {report_path}")


if __name__ == "__main__":
    main()
