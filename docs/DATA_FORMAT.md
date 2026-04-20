# Data Format Guide

This document describes the data format required for fine-tuning XTTS-v2, and how to prepare your own Arabic speech data.

## Required Directory Structure

```
data/
└── <speaker_or_dialect>/
    ├── metadata_train.csv      # Training split
    ├── metadata_eval.csv       # Evaluation split
    └── wavs/                   # Audio files
        ├── clip_00001.wav
        ├── clip_00002.wav
        └── ...
```

## Metadata CSV Format

The CSV files use **pipe `|` as delimiter** (not comma) and **must include a header row**.

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `audio_file` | string | Relative path to audio file (e.g., `wavs/clip_001.wav`) |
| `text` | string | Arabic transcript of the audio |
| `speaker_name` | string | Speaker or dialect identifier (e.g., `"sudanese"`, `"egyptian"`) |

### Example

```csv
audio_file|text|speaker_name
wavs/clip_00001.wav|السلام عليكم ورحمة الله وبركاته|sudanese
wavs/clip_00002.wav|كيف حالك اليوم|sudanese
wavs/clip_00003.wav|الذكاء الاصطناعي يتطور بسرعة كبيرة|sudanese
```

### Optional Column

| Column | Type | Description |
|--------|------|-------------|
| `emotion_name` | string | Emotion label (defaults to `"neutral"` if omitted) |

## Audio Requirements

| Requirement | Value | Notes |
|-------------|-------|-------|
| **Sample Rate** | 22,050 Hz | Auto-resampled if different (16kHz, 44.1kHz, etc. all work) |
| **Channels** | Mono | Auto-converted from stereo if needed |
| **Format** | WAV | MP3 and FLAC also supported, WAV preferred |
| **Duration** | 0.5 – 11.6 seconds | Clips outside this range are skipped during training |
| **Recommended Duration** | 2 – 8 seconds | Best quality range for XTTS-v2 |
| **Amplitude** | [-1.0, 1.0] | Auto-clipped if out of range |
| **Text Length** | Max 200 characters | Per clip |
| **Minimum Dataset** | ~2 minutes total | More data = better quality; 1–10 hours recommended |

## Train/Eval Split

- **Default**: 85% training / 15% evaluation
- If you provide both `metadata_train.csv` and `metadata_eval.csv`, they are used as-is
- If you only have one file, the training script can split automatically

## Text Guidelines

### For Dialect Output (Recommended)

- **Do NOT add tashkeel** (diacritics) — let the model learn pronunciation from the audio
- **Do** correct hamza placement (أ, إ, آ) for tokenizer consistency
- **Do** spell out numbers as words (70% → سبعون بالمئة)
- **Do** expand symbols (&→و, $→دولار)
- Write text in the dialect as spoken, not in formal MSA

### For Formal MSA Output

- Same as above, but you may optionally add tashkeel
- Use our preprocessor: `python scripts/arabic_preprocessor.py`

## Preparing Data from HuggingFace Datasets

Many Arabic speech datasets are available on HuggingFace. Here's how to convert them to XTTS-v2 format:

### Example: Converting a HuggingFace Dataset

```python
"""
Convert a HuggingFace audio dataset to XTTS-v2 training format.

Usage:
    python scripts/prepare_hf_dataset.py \
        --dataset "MAdel121/arabic-egy-cleaned" \
        --speaker "egyptian" \
        --output data/egyptian \
        --max-samples 5000
"""
import os
import soundfile as sf
from datasets import load_dataset

dataset = load_dataset("MAdel121/arabic-egy-cleaned", split="train")

output_dir = "data/egyptian"
wavs_dir = os.path.join(output_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

rows = []
for i, sample in enumerate(dataset):
    # Save audio
    audio = sample["audio"]
    wav_path = f"wavs/clip_{i:06d}.wav"
    sf.write(
        os.path.join(output_dir, wav_path),
        audio["array"],
        audio["sampling_rate"],  # XTTS auto-resamples
    )
    rows.append(f"{wav_path}|{sample['text']}|egyptian")

# Write metadata (90/10 split)
split_idx = int(len(rows) * 0.9)
with open(os.path.join(output_dir, "metadata_train.csv"), "w") as f:
    f.write("audio_file|text|speaker_name\n")
    f.write("\n".join(rows[:split_idx]))

with open(os.path.join(output_dir, "metadata_eval.csv"), "w") as f:
    f.write("audio_file|text|speaker_name\n")
    f.write("\n".join(rows[split_idx:]))

print(f"Saved {len(rows)} clips to {output_dir}")
```

## Quality Checklist

Before training, verify your data meets these criteria:

- [ ] All audio files exist and are playable
- [ ] Audio duration is between 0.5 and 11.6 seconds per clip
- [ ] Text accurately matches the spoken audio
- [ ] Text length is under 200 characters per clip
- [ ] No silence-only clips
- [ ] CSV uses pipe `|` delimiter with header row
- [ ] Speaker name is consistent across all rows (for single-speaker fine-tuning)
- [ ] Minimum ~2 minutes of total audio (1+ hours recommended)

## Data Sources for Arabic TTS

### Fully Free

| Dataset | Dialect | Size | Sample Rate | Best For |
|---------|---------|------|-------------|----------|
| [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) | Egyptian | 72h / 103K clips | 16 kHz | Large-scale Egyptian fine-tuning |
| [Kaggle Egyptian Audio](https://www.kaggle.com/datasets/ahmedshafiq12/egyptian-audio-dataset-collected-from-youtube) | Egyptian | Unknown | Unknown | Supplementary Egyptian data |
| [MagicHub ASR-EgArbCSC](https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/) | Egyptian | 5.5h / 4 speakers | 16 kHz | Small experiments only |

### Free with Registration

| Dataset | Dialect | Size | Sample Rate | Best For |
|---------|---------|------|-------------|----------|
| [MGB-3](https://arabicspeech.org/mgb3-asr/) | Egyptian | 15h+ | Unknown | Quality Egyptian data |
| [Mozilla Common Voice](https://commonvoice.mozilla.org) | Mixed MSA | ~300h | 48 kHz | MSA fine-tuning, highest sample rate |
| [ADI-5](https://github.com/ARBML/klaam) | Multi-dialect | 50h+ | Unknown | Multi-dialect experiments |

### Other Resources

| Dataset | Dialect | Notes |
|---------|---------|-------|
| [OpenSLR #46](https://openslr.org/46/) | Tunisian MSA | Tunisian Modern Standard Arabic |
| [OpenSLR #108](https://openslr.org/108/) | Multi (includes Arabic) | Media/broadcast speech |
| [OpenSLR #132](https://openslr.org/132/) | Quranic | Recitation — not conversational |
| Custom YouTube collection | Any dialect | Use yt-dlp + Whisper for transcription |

> **Tip**: For the best dialect-specific results, collect data from a single speaker or a small group of speakers with the same dialect. Mixed-speaker datasets work but produce less distinctive voices.
>
> **Sample rate matters**: 22kHz+ source data preserves more vocal detail. If choosing between datasets of similar quality, prefer the one with the higher sample rate.
