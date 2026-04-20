"""
Improved Arabic inference pipeline for XTTS-v2.

Improvements over baseline:
  1. Arabic text preprocessing (hamza, numbers, symbols)
  2. Smart text chunking (sentence-aware, respects char limit)
  3. Optimized generation parameters for Arabic
  4. Post-processing (pause compression, rambling detection, silence trim)
  5. Optional tashkeel for formal MSA mode

Usage:
    conda activate new-arabic-tts
    python scripts/infer.py --text "مرحباً بكم" --output outputs/test.wav
    python scripts/infer.py --text-file input.txt --output outputs/test.wav
    python scripts/infer.py --text "مرحباً" --tashkeel  # formal MSA mode
"""

import argparse
import os
import re
import sys
import time
import json

import numpy as np
import torch
import soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from scripts.arabic_preprocessor import ArabicPreprocessor

# --- Constants ---
SAMPLE_RATE = 24000
MAX_CHARS_PER_CHUNK = 160  # XTTS-v2 Arabic char limit safety margin
MIN_CHUNK_CHARS = 30       # Don't split below this

# Arabic sentence-ending punctuation
SENTENCE_ENDINGS = re.compile(r"[.!?؟。！？]+")
# Arabic comma and semicolon for secondary splits
SECONDARY_SPLITS = re.compile(r"[،؛,;]+")


# --- Text Chunking ---

def split_into_sentences(text):
    """Split text into sentences at Arabic punctuation marks."""
    parts = SENTENCE_ENDINGS.split(text)
    delimiters = SENTENCE_ENDINGS.findall(text)

    sentences = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        # Re-attach the punctuation
        if i < len(delimiters):
            part = part + delimiters[i]
        sentences.append(part)
    return sentences


def split_long_sentence(sentence, max_chars=MAX_CHARS_PER_CHUNK):
    """Split a long sentence at commas/semicolons, then by word boundary."""
    if len(sentence) <= max_chars:
        return [sentence]

    # Try splitting at commas/semicolons first
    parts = SECONDARY_SPLITS.split(sentence)
    if len(parts) > 1:
        chunks = []
        current = ""
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if current and len(current) + len(part) + 2 > max_chars:
                chunks.append(current.strip())
                current = part
            else:
                current = (current + "، " + part).strip("، ") if current else part
        if current.strip():
            chunks.append(current.strip())

        # Recursively split any still-too-long chunks
        result = []
        for chunk in chunks:
            result.extend(split_long_sentence(chunk, max_chars))
        return result

    # Last resort: split at word boundary
    words = sentence.split()
    chunks = []
    current = ""
    for word in words:
        if current and len(current) + len(word) + 1 > max_chars:
            chunks.append(current.strip())
            current = word
        else:
            current = current + " " + word if current else word
    if current.strip():
        chunks.append(current.strip())
    return chunks


def merge_short_chunks(chunks, min_chars=MIN_CHUNK_CHARS):
    """Merge very short chunks with their neighbors."""
    if len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        if len(merged[-1]) < min_chars or len(chunk) < min_chars:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)
    return merged


def chunk_text(text, max_chars=MAX_CHARS_PER_CHUNK):
    """Full text chunking pipeline."""
    sentences = split_into_sentences(text)
    chunks = []
    for sentence in sentences:
        chunks.extend(split_long_sentence(sentence, max_chars))
    chunks = merge_short_chunks(chunks)
    return chunks


# --- Post-Processing ---

def compress_pauses(wav, sr, max_silence_ms=150):
    """Cap internal silence duration to max_silence_ms."""
    max_silence_samples = int(sr * max_silence_ms / 1000)
    threshold = 0.01

    # Find silent regions
    is_silent = np.abs(wav) < threshold
    result = []
    silence_count = 0

    for i, sample in enumerate(wav):
        if is_silent[i]:
            silence_count += 1
            if silence_count <= max_silence_samples:
                result.append(sample)
        else:
            silence_count = 0
            result.append(sample)

    return np.array(result, dtype=wav.dtype)


def trim_trailing_silence(wav, sr, threshold=0.01, keep_ms=50):
    """Trim silence from end, keeping a small tail."""
    keep_samples = int(sr * keep_ms / 1000)
    end = len(wav)
    while end > 0 and abs(wav[end - 1]) < threshold:
        end -= 1
    end = min(len(wav), end + keep_samples)
    return wav[:end]


def detect_and_trim_rambling(wav, sr, expected_duration=None):
    """Detect over-generation (rambling) and trim at the nearest silence."""
    if expected_duration is None:
        return wav

    actual_duration = len(wav) / sr
    if actual_duration <= expected_duration * 1.5:
        return wav  # within acceptable range

    # Find a good cut point near the expected duration
    target_sample = int(expected_duration * 1.2 * sr)
    search_window = int(sr * 0.5)  # search 0.5s around target
    start = max(0, target_sample - search_window)
    end = min(len(wav), target_sample + search_window)

    # Find quietest point in window
    window = np.abs(wav[start:end])
    frame_size = int(sr * 0.025)
    min_energy = float("inf")
    cut_point = target_sample

    for i in range(0, len(window) - frame_size, frame_size // 2):
        energy = np.mean(window[i : i + frame_size])
        if energy < min_energy:
            min_energy = energy
            cut_point = start + i

    # Apply fade out
    fade_samples = int(sr * 0.05)
    result = wav[:cut_point + fade_samples].copy()
    fade = np.linspace(1.0, 0.0, fade_samples)
    result[-fade_samples:] *= fade

    return result


def apply_sentence_taper(wav, sr, taper_ms=400):
    """Apply gentle amplitude taper at the end of a sentence."""
    taper_samples = min(int(sr * taper_ms / 1000), len(wav) // 4)
    result = wav.copy()
    taper = np.linspace(1.0, 0.0, taper_samples) ** 2  # quadratic fade
    result[-taper_samples:] *= taper
    return result


def post_process_chunk(wav, sr):
    """Full post-processing pipeline for a single chunk."""
    wav = compress_pauses(wav, sr, max_silence_ms=150)
    wav = trim_trailing_silence(wav, sr)
    wav = apply_sentence_taper(wav, sr, taper_ms=400)
    return wav


# --- Main Inference ---

def load_model(model_dir):
    """Load XTTS-v2 model."""
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir)
    model.cuda()
    model.eval()
    return model


def load_speaker(model_dir, speaker_name):
    """Load a built-in speaker embedding."""
    speakers = torch.load(
        os.path.join(model_dir, "speakers_xtts.pth"), weights_only=False
    )
    if speaker_name not in speakers:
        available = ", ".join(sorted(speakers.keys()))
        raise ValueError(f"Speaker '{speaker_name}' not found. Available: {available}")
    speaker_data = speakers[speaker_name]
    return (
        speaker_data["gpt_cond_latent"].cuda(),
        speaker_data["speaker_embedding"].cuda(),
    )


def generate(
    model,
    text,
    gpt_cond_latent,
    speaker_embedding,
    preprocessor=None,
    temperature=0.3,
    top_p=0.7,
    repetition_penalty=10.0,
    sentence_pause=0.35,
    paragraph_pause=0.7,
    seed=12345,
):
    """
    Generate Arabic speech with preprocessing, chunking, and post-processing.

    Args:
        model: Loaded XTTS-v2 model.
        text: Input Arabic text (can be multi-sentence/paragraph).
        gpt_cond_latent: Speaker style conditioning.
        speaker_embedding: Speaker identity embedding.
        preprocessor: ArabicPreprocessor instance (created if None).
        temperature: Generation temperature (lower = more deterministic).
        top_p: Nucleus sampling threshold.
        repetition_penalty: Penalty for repeated tokens.
        sentence_pause: Pause between sentences in seconds.
        paragraph_pause: Pause at end of paragraphs in seconds.
        seed: Random seed for reproducibility.

    Returns:
        dict with 'wav' (numpy array), 'sr' (sample rate), 'stats' (generation info).
    """
    if preprocessor is None:
        preprocessor = ArabicPreprocessor()

    # Preprocess
    processed_text = preprocessor.process(text)

    # Chunk
    chunks = chunk_text(processed_text)

    # Generate
    all_wav = []
    stats = {"chunks": [], "total_generation_time": 0}
    pause = np.zeros(int(SAMPLE_RATE * sentence_pause))
    final_pause = np.zeros(int(SAMPLE_RATE * paragraph_pause))

    for i, chunk in enumerate(chunks):
        # Set deterministic seed per chunk
        torch.manual_seed(seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + i)

        t0 = time.time()
        out = model.inference(
            text=chunk,
            language="ar",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        gen_time = time.time() - t0

        wav = out["wav"]
        wav = post_process_chunk(wav, SAMPLE_RATE)

        chunk_stats = {
            "index": i,
            "text": chunk,
            "chars": len(chunk),
            "duration_s": round(len(wav) / SAMPLE_RATE, 2),
            "generation_time_s": round(gen_time, 2),
        }
        stats["chunks"].append(chunk_stats)
        stats["total_generation_time"] += gen_time

        all_wav.append(wav)
        # Add pause between chunks (longer for last sentence)
        if i < len(chunks) - 1:
            all_wav.append(pause)

    # Concatenate and add final pause
    final_wav = np.concatenate(all_wav)
    final_wav = np.concatenate([final_wav, final_pause])

    stats["total_duration_s"] = round(len(final_wav) / SAMPLE_RATE, 2)
    stats["total_generation_time"] = round(stats["total_generation_time"], 2)
    stats["rtf"] = round(
        stats["total_generation_time"] / stats["total_duration_s"], 3
    )
    stats["num_chunks"] = len(chunks)
    stats["params"] = {
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "sentence_pause": sentence_pause,
        "seed": seed,
    }

    return {"wav": final_wav, "sr": SAMPLE_RATE, "stats": stats}


def main():
    parser = argparse.ArgumentParser(description="Arabic TTS Inference")
    parser.add_argument("--text", type=str, help="Arabic text to synthesize")
    parser.add_argument("--text-file", type=str, help="Read text from file")
    parser.add_argument("--output", type=str, required=True, help="Output WAV path")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "base"),
        help="Model directory",
    )
    parser.add_argument("--speaker", type=str, default="Gilberto Mathias")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--rep-penalty", type=float, default=10.0)
    parser.add_argument("--pause", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--tashkeel", action="store_true",
        help="Enable tashkeel for formal MSA pronunciation",
    )
    args = parser.parse_args()

    if not args.text and not args.text_file:
        parser.error("Provide --text or --text-file")

    text = args.text
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read().strip()

    # Load model
    print(f"Loading model from {args.model_dir}...")
    model = load_model(args.model_dir)

    # Load speaker
    print(f"Loading speaker: {args.speaker}")
    gpt_cond_latent, speaker_embedding = load_speaker(args.model_dir, args.speaker)

    # Create preprocessor
    preprocessor = ArabicPreprocessor(enable_tashkeel=args.tashkeel)
    mode = "MSA (tashkeel)" if args.tashkeel else "Dialect (no tashkeel)"
    print(f"Mode: {mode}")

    # Generate
    print(f"Generating...")
    result = generate(
        model=model,
        text=text,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        preprocessor=preprocessor,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
        sentence_pause=args.pause,
        seed=args.seed,
    )

    # Save output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sf.write(args.output, result["wav"], result["sr"])

    # Print stats
    stats = result["stats"]
    print(f"\nResult:")
    print(f"  Output:     {args.output}")
    print(f"  Duration:   {stats['total_duration_s']}s")
    print(f"  Gen Time:   {stats['total_generation_time']}s")
    print(f"  RTF:        {stats['rtf']}")
    print(f"  Chunks:     {stats['num_chunks']}")

    for chunk in stats["chunks"]:
        print(f"  [{chunk['index']+1}] {chunk['duration_s']:5.2f}s | {chunk['text'][:50]}...")

    # Save stats alongside wav
    stats_path = args.output.replace(".wav", "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
