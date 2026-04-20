"""
Fine-tune XTTS-v2 on curated Egyptian Arabic data.

Uses the cleaned dataset from Phase 5 (data/egyptian/) to fine-tune
the GPT component of XTTS-v2. The base model weights are used as a
starting point, and only the GPT layers are updated.

Training configuration:
  - 4 epochs (conservative to avoid overfitting on 5h of data)
  - Batch size 4, gradient accumulation 2 (effective batch = 8)
  - Learning rate 5e-6 (AdamW)
  - fp32 training (fp16/mixed precision causes NaN losses with XTTS GPT)
  - Saves best checkpoint + every 1000 steps

Usage:
    conda activate new-arabic-tts
    python scripts/train.py

Output:
    models/finetuned/run/training/   (checkpoints, logs, config)
"""

import os
import gc
import sys
import json
import time
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "egyptian"
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "base"
OUTPUT_DIR = PROJECT_ROOT / "models" / "finetuned"

TRAIN_CSV = str(DATA_DIR / "metadata_train.csv")
EVAL_CSV = str(DATA_DIR / "metadata_eval.csv")

# --- Training Config ---
LANGUAGE = "ar"
NUM_EPOCHS = 4
BATCH_SIZE = 4
GRAD_ACCUM = 2
LEARNING_RATE = 5e-6
MAX_AUDIO_LENGTH = 255995  # ~11.6 seconds at 22050 Hz
SAVE_STEP = 1000


def main():
    print("=" * 70)
    print("  XTTS-v2 Fine-Tuning — Egyptian Arabic")
    print("=" * 70)
    t_start = time.time()

    OUT_PATH = str(OUTPUT_DIR / "run" / "training")
    os.makedirs(OUT_PATH, exist_ok=True)

    # --- Download DVAE and mel norm files ---
    CHECKPOINTS_OUT = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")
    os.makedirs(CHECKPOINTS_OUT, exist_ok=True)

    DVAE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
    MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT, "dvae.pth")
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT, "mel_stats.pth")

    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        print("[1/4] Downloading DVAE files...")
        ModelManager._download_model_files([MEL_NORM_LINK, DVAE_LINK], CHECKPOINTS_OUT, progress_bar=True)
    else:
        print("[1/4] DVAE files already downloaded")

    # --- Use local base model files ---
    TOKENIZER_FILE = str(BASE_MODEL_DIR / "vocab.json")
    XTTS_CHECKPOINT = str(BASE_MODEL_DIR / "model.pth")
    XTTS_CONFIG_FILE = str(BASE_MODEL_DIR / "config.json")

    print(f"[2/4] Base model: {BASE_MODEL_DIR}")
    print(f"      Train CSV:  {TRAIN_CSV}")
    print(f"      Eval CSV:   {EVAL_CSV}")

    # --- Dataset config ---
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="egyptian_arabic",
        path=str(DATA_DIR),
        meta_file_train=TRAIN_CSV,
        meta_file_val=EVAL_CSV,
        language=LANGUAGE,
    )

    # --- Model args ---
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6 seconds
        min_conditioning_length=66150,    # 3 seconds
        debug_loading_failures=False,
        max_wav_length=MAX_AUDIO_LENGTH,
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # --- Trainer config ---
    config = GPTTrainerConfig(
        epochs=NUM_EPOCHS,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="GPT_XTTS_AR_FT",
        project_name="Arabic_TTS",
        run_description="Fine-tuning XTTS-v2 GPT on Egyptian Arabic (5h, single speaker)",
        dashboard_logger="tensorboard",
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=SAVE_STEP,
        save_n_checkpoints=3,
        save_checkpoints=True,
        print_eval=False,
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LEARNING_RATE,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [50000 * 18, 150000 * 18, 300000 * 18],
            "gamma": 0.5,
            "last_epoch": -1,
        },
        test_sentences=[],
    )

    # --- Init model ---
    print("[3/4] Initializing model...")
    model = GPTTrainer.init_from_config(config)

    # --- Load data ---
    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    print(f"      Train samples: {len(train_samples)}")
    print(f"      Eval samples:  {len(eval_samples)}")

    # --- Train ---
    print(f"[4/4] Starting training...")
    print(f"      Epochs:     {NUM_EPOCHS}")
    print(f"      Batch size: {BATCH_SIZE} (x{GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective)")
    print(f"      LR:         {LEARNING_RATE}")
    print(f"      Save every: {SAVE_STEP} steps")
    print(f"      Output:     {OUT_PATH}")
    print()

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACCUM,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    elapsed = (time.time() - t_start) / 3600
    print(f"\n{'='*70}")
    print(f"  Training Complete!")
    print(f"  Total time: {elapsed:.1f} hours")
    print(f"  Output: {trainer.output_path}")
    print(f"{'='*70}")

    # Save training summary
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "dataset": "egyptian_arabic (MAdel121/arabic-egy-cleaned)",
        "train_clips": len(train_samples),
        "eval_clips": len(eval_samples),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "learning_rate": LEARNING_RATE,
        "training_hours": round(elapsed, 2),
        "output_path": trainer.output_path,
        "base_model": str(BASE_MODEL_DIR),
    }
    summary_path = PROJECT_ROOT / "docs" / "benchmarks" / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    del model, trainer, train_samples, eval_samples
    gc.collect()


if __name__ == "__main__":
    main()
