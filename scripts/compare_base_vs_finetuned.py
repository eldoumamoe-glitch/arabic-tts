"""
Generate comparison outputs: base model vs finetuned model.

Uses the same 10 AI-themed Arabic test sentences from baseline_test.py.
Produces two WAV files in outputs/ for side-by-side listening.

The finetuned model uses reference audio from the training data for
speaker conditioning, ensuring the voice identity matches the trained
speaker instead of drifting to a random built-in voice.

Usage:
    conda activate new-arabic-tts
    python scripts/compare_base_vs_finetuned.py
"""

import os
import sys
import time
import glob
import json
import numpy as np
import torch
import soundfile as sf
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.arabic_preprocessor import ArabicPreprocessor

# --- Config ---
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "base"
FINETUNED_DIR = PROJECT_ROOT / "models" / "finetuned"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SPEAKER_NAME = "Gilberto Mathias"  # used for base model only
SAMPLE_RATE = 24000
SENTENCE_PAUSE = 0.35

# Reference audio clips from training data (for finetuned model conditioning)
REF_AUDIO_PATHS = [
    str(PROJECT_ROOT / "data" / "Egyption" / "clean" / "wavs" / "ep001_0057.wav"),
    str(PROJECT_ROOT / "data" / "Egyption" / "clean" / "wavs" / "ep001_0060.wav"),
    str(PROJECT_ROOT / "data" / "Egyption" / "clean" / "wavs" / "ep001_0064.wav"),
]

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


def find_best_checkpoint():
    """Find the best_model.pth from the latest training run."""
    # Look for best_model.pth under finetuned dir
    pattern = str(FINETUNED_DIR / "run" / "training" / "GPT_XTTS_AR_FT*" / "best_model.pth")
    matches = glob.glob(pattern)
    if matches:
        # Pick most recent
        return max(matches, key=os.path.getmtime)

    # Fallback: any best_model.pth
    pattern = str(FINETUNED_DIR / "**" / "best_model.pth")
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return max(matches, key=os.path.getmtime)

    return None


def load_base_model():
    """Load the base XTTS-v2 model."""
    config = XttsConfig()
    config.load_json(str(BASE_MODEL_DIR / "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=str(BASE_MODEL_DIR))
    model.cuda()
    model.eval()
    return model


def load_finetuned_model(checkpoint_path):
    """Load finetuned model using base config + finetuned checkpoint."""
    config = XttsConfig()
    config.load_json(str(BASE_MODEL_DIR / "config.json"))
    model = Xtts.init_from_config(config)
    # Load base checkpoint first, then override with finetuned GPT weights
    model.load_checkpoint(config, checkpoint_dir=str(BASE_MODEL_DIR))
    # Load the finetuned GPT weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        model_state = checkpoint["model"]
    else:
        model_state = checkpoint
    # Strip 'xtts.' prefix from checkpoint keys (GPTTrainer saves with this prefix)
    stripped_state = {}
    for k, v in model_state.items():
        new_key = k.replace("xtts.", "", 1) if k.startswith("xtts.") else k
        stripped_state[new_key] = v
    # Load only the matching keys
    model_dict = model.state_dict()
    filtered = {k: v for k, v in stripped_state.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    print(f"    Loaded {len(filtered)}/{len(stripped_state)} finetuned weights")
    model.cuda()
    model.eval()
    return model


def generate_all_sentences(model, gpt_cond_latent, speaker_embedding, label):
    """Generate all test sentences and concatenate."""
    preprocessor = ArabicPreprocessor()
    pause = np.zeros(int(SAMPLE_RATE * SENTENCE_PAUSE))
    all_wav = []
    total_gen_time = 0

    for i, text in enumerate(TEST_SENTENCES):
        processed = preprocessor.process(text)
        torch.manual_seed(12345 + i)
        torch.cuda.manual_seed(12345 + i)

        t0 = time.time()
        out = model.inference(
            text=processed,
            language="ar",
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            **GENERATION_PARAMS,
        )
        gen_time = time.time() - t0
        total_gen_time += gen_time

        wav = out["wav"]
        dur = len(wav) / SAMPLE_RATE
        print(f"    [{i+1:2d}/10] {dur:5.2f}s (gen {gen_time:.1f}s) | {text[:50]}...")

        all_wav.append(wav)
        if i < len(TEST_SENTENCES) - 1:
            all_wav.append(pause)

    final_wav = np.concatenate(all_wav)
    total_dur = len(final_wav) / SAMPLE_RATE
    print(f"    Total: {total_dur:.1f}s audio, {total_gen_time:.1f}s generation")
    return final_wav


def main():
    print("=" * 70)
    print("  Base vs Finetuned Comparison")
    print("=" * 70)

    # Find finetuned checkpoint
    ckpt = find_best_checkpoint()
    if ckpt is None:
        print("ERROR: No finetuned checkpoint found!")
        print(f"  Searched: {FINETUNED_DIR}")
        sys.exit(1)
    print(f"\nFinetuned checkpoint: {ckpt}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Base model ---
    print("\n[1/4] Loading base model...")
    base_model = load_base_model()

    # Base model uses built-in speaker embedding
    print(f"  Loading built-in speaker: {SPEAKER_NAME}")
    speakers = torch.load(str(BASE_MODEL_DIR / "speakers_xtts.pth"), weights_only=False)
    speaker_data = speakers[SPEAKER_NAME]
    base_gpt_cond = speaker_data["gpt_cond_latent"].cuda()
    base_spk_emb = speaker_data["speaker_embedding"].cuda()

    print("\n[2/4] Generating with BASE model...")
    base_wav = generate_all_sentences(base_model, base_gpt_cond, base_spk_emb, "base")
    base_path = str(OUTPUT_DIR / "base_model_test.wav")
    sf.write(base_path, base_wav, SAMPLE_RATE)
    print(f"    Saved: {base_path}")

    # Free memory
    del base_model, base_gpt_cond, base_spk_emb
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # --- Finetuned model ---
    print("\n[3/4] Loading finetuned model...")
    ft_model = load_finetuned_model(ckpt)

    # Finetuned model uses reference audio from training data for
    # consistent voice identity (this is the key fix for voice drift)
    ref_existing = [p for p in REF_AUDIO_PATHS if os.path.exists(p)]
    if not ref_existing:
        print("  WARNING: No reference audio found, falling back to built-in speaker")
        speakers = torch.load(str(BASE_MODEL_DIR / "speakers_xtts.pth"), weights_only=False)
        speaker_data = speakers[SPEAKER_NAME]
        ft_gpt_cond = speaker_data["gpt_cond_latent"].cuda()
        ft_spk_emb = speaker_data["speaker_embedding"].cuda()
    else:
        print(f"  Computing speaker conditioning from {len(ref_existing)} reference clip(s)...")
        ft_gpt_cond, ft_spk_emb = ft_model.get_conditioning_latents(
            audio_path=ref_existing
        )
        ft_gpt_cond = ft_gpt_cond.cuda()
        ft_spk_emb = ft_spk_emb.cuda()

    print("\n[4/4] Generating with FINETUNED model...")
    ft_wav = generate_all_sentences(ft_model, ft_gpt_cond, ft_spk_emb, "finetuned")
    ft_path = str(OUTPUT_DIR / "finetuned_model_test.wav")
    sf.write(ft_path, ft_wav, SAMPLE_RATE)
    print(f"    Saved: {ft_path}")

    del ft_model, ft_gpt_cond, ft_spk_emb
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("  Comparison Complete!")
    print(f"  Base output:      {base_path}")
    print(f"  Finetuned output: {ft_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
