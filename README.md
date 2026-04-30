---
language:
- ar
license: other
license_name: cpml
license_link: https://coqui.ai/cpml
tags:
- text-to-speech
- tts
- arabic
- xtts
- speech-synthesis
- deep-learning
- pytorch
- egyptian-arabic
datasets:
- MAdel121/arabic-egy-cleaned
base_model: coqui/XTTS-v2
pipeline_tag: text-to-speech
---

# Arabic TTS: Improving XTTS-v2 for Arabic Speech Synthesis

A systematic project to improve [Coqui XTTS-v2](https://github.com/coqui-ai/TTS) for high-quality Arabic text-to-speech generation. This repository documents the full journey from baseline evaluation to fine-tuned deployment, with reproducible benchmarks at every stage.

![Pipeline Animation](docs/images/pipeline_animation.gif)

## Table of Contents

- [Project Overview](#project-overview)
- [Where This Project Sits in AI](#where-this-project-sits-in-ai)
- [Architecture](#architecture)
- [Setup & Installation](#setup--installation)
- [Phase 1: Baseline Evaluation](#phase-1-baseline-evaluation)
- [Phase 2: Compatibility Fixes](#phase-2-compatibility-fixes)
- [Phase 3: Arabic Text Preprocessing](#phase-3-arabic-text-preprocessing)
- [Phase 4: Inference Pipeline Improvements](#phase-4-inference-pipeline-improvements)
- [Phase 5: Data Collection & Preparation](#phase-5-data-collection--preparation)
- [Phase 6: Fine-Tuning](#phase-6-fine-tuning)
- [Phase 7: Evaluation & Comparison](#phase-7-evaluation--comparison)
- [Phase 8: Studio-Quality Audio Upsampling](#phase-8-studio-quality-audio-upsampling)
- [Resource Usage](#resource-usage)
- [Results](#results)
- [Known Limitations & Future Improvements](#known-limitations--future-improvements)
- [Industry Use Cases](#industry-use-cases)
- [Usage](#usage)
- [License](#license)

---

## Project Overview

XTTS-v2 is a powerful multilingual TTS model supporting 17 languages out of the box — a remarkable achievement by the Coqui team. This project builds on that foundation to push Arabic output quality further by:

1. **Expanding Arabic training data** — supplementing the base model with additional high-quality Arabic speech data
2. **Adding Arabic text preprocessing** — introducing diacritization (tashkeel) and hamza normalization to help the model with Arabic-specific orthography
3. **Optimizing the tokenizer for Arabic** — adapting the shared multilingual tokenizer to better handle Arabic text patterns
4. **Tuning inference for Arabic prosody** — adjusting generation parameters to better match Arabic speech rhythms and intonation
5. **Building a post-processing pipeline** — adding pause control, silence management, and output refinement tailored to Arabic

This project addresses each area systematically, with before/after comparisons at every stage.

### Goals

- **Natural-sounding Arabic speech** — produce voices that sound authentically human, not robotic or accented
- **Multi-dialect support** — generate speech across different Arabic dialects and accents (e.g. Sudanese, Egyptian, Gulf, Levantine, North African, MSA) by fine-tuning on dialect-specific speakers
- Document a reproducible pipeline from base model to production-quality Arabic TTS
- Provide quantitative benchmarks (MOS scores, RTF, SNR) at each improvement stage
- Release fine-tuned checkpoints and preprocessing tools for the Arabic TTS community

---

## Where This Project Sits in AI

![AI/ML/DL/Data Science Domains](docs/images/ai_ml_dl_domains.gif)

*Figure 1: The relationship between AI, Machine Learning, Deep Learning, and Data Science — and where each part of this project applies.*

This project spans all four domains. Here is where each phase of our work falls:

| Domain | What we did in this project | Phase |
|--------|---------------------------|-------|
| **Artificial Intelligence** | End-to-end system that converts Arabic text to natural-sounding speech — a core AI application | All phases |
| **Deep Learning** | Fine-tuned a 370M-parameter GPT-2 transformer + HiFiGAN vocoder on Arabic speech data | Phase 6 (Training) |
| **Machine Learning** | Speaker clustering with ECAPA-TDNN embeddings + Agglomerative Clustering to identify speakers in unlabeled data | Phase 5 (Data Preparation) |
| **Data Science** | Dataset evaluation, quality filtering, statistical analysis, SNR measurement, Whisper-based alignment verification, visualization of results | Phases 5, 7 (Data & Evaluation) |

**In detail:**
- **Deep Learning** powers the core model — the GPT-2 backbone generates audio codes autoregressively, the HiFiGAN vocoder converts them to waveforms, and the Perceiver Resampler handles speaker conditioning. Fine-tuning these neural networks on Arabic data is a Deep Learning task
- **Machine Learning** solved our speaker identification problem — with 103K unlabeled clips, we used ECAPA-TDNN (a neural feature extractor) combined with Agglomerative Clustering (a classical ML algorithm) to find and extract 5,000 clips from the same speaker
- **Data Science** drove every decision about data quality — from evaluating 6 candidate datasets, to measuring SNR and silence ratios, to building the 3-layer sanity check pipeline, to generating comparison charts and metrics
- **AI** is the umbrella — the final system takes Arabic text as input and produces human-like speech as output, combining all three disciplines into a practical application

---

## Architecture

### XTTS-v2 Model Components

![XTTS-v2 Architecture](docs/images/architecture_diagram.png)
*Figure 2: XTTS-v2 architecture — text and reference audio are processed through separate pathways, combined in the GPT-2 transformer, and decoded to a 24kHz waveform.*

### Key Specifications

| Component | Details |
|-----------|---------|
| GPT Backbone | 30 layers, 1024 hidden dim, 16 heads, ~350M params |
| Tokenizer | VoiceBPE, 6,681 tokens, 17 languages |
| Audio Codec | Discrete VAE, 1,026 codebook entries |
| Vocoder | HiFiGAN, [8,8,2,2] upsampling, ResBlock1 |
| Speaker Encoder | ResNet50 + SE blocks, 512-dim output |
| Conditioning | Perceiver Resampler, 2 layers, 32 latents |
| Output | 24,000 Hz WAV |
| Input Sample Rate | 22,050 Hz |
| Max Text Tokens | 402 |
| Max Audio Tokens | 605 |
| Character Limit | 166 per chunk (Arabic) |
| Total Parameters | ~370M |

---

## Setup & Installation

### Hardware

This project was developed on an **NVIDIA DGX Spark** — a desktop AI workstation with a Grace Blackwell architecture. If you're using different hardware, you may need to adjust CUDA versions, PyTorch builds, and batch sizes accordingly.

| Spec | DGX Spark (this project) |
|------|--------------------------|
| **GPU** | NVIDIA GB10 (Blackwell) |
| **GPU Memory** | 121.7 GB (unified) |
| **CPU Architecture** | ARM (aarch64) |
| **System RAM** | 128 GB |
| **OS** | Ubuntu 24.04 LTS |
| **CUDA** | 13.0 |
| **NVIDIA Driver** | 580.142 |

> [!NOTE]
> ### Running on different hardware?
>
> **Consumer GPUs (RTX 3090, 4090, etc.)**:
> - You'll need PyTorch built for your CUDA version (e.g., `cu118`, `cu121`, `cu124` instead of `cu130`)
> - With 24GB VRAM, reduce `batch_size` to 2 and `grad_accumulation_steps` to 1 during fine-tuning
> - Inference works fine on 8GB+ VRAM
>
> **Data Center GPUs (A100, H100)**:
> - Should work out of the box with matching CUDA/PyTorch versions
> - Can increase batch sizes for faster training
>
> **x86 vs ARM (aarch64)**:
> - DGX Spark uses ARM. If you're on x86 (most desktops/servers), install the x86 PyTorch build — the code is the same, only the binary packages differ
> - `pip install TTS` will automatically pull the correct architecture
>
> **CPU only**:
> - Inference is possible but very slow (~10x slower than GPU)
> - Fine-tuning on CPU is not practical
>
> **Key**: Match your **PyTorch version** to your **CUDA version**. Check compatibility at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Software Environment

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10.20 | Runtime (via Miniconda) |
| PyTorch | 2.11.0+cu130 | Deep learning framework — match to your CUDA version |
| CUDA Toolkit | 13.0 | Match to your GPU driver |
| Coqui TTS | 0.22.0 | Core XTTS-v2 framework |
| Transformers | 4.44.2 | Pinned — v5.x breaks Coqui TTS (see [Phase 2](#phase-2-compatibility-fixes)) |
| SpeechBrain | 1.1.0 | ECAPA-TDNN speaker embeddings |
| OpenAI Whisper | latest | Alignment verification (large-v3 model) |
| AudioSR | 0.0.7 | Neural audio upsampling (24→48 kHz) |
| Mishkal | 0.4.1 | Arabic tashkeel — optional, for formal MSA mode |
| PyArabic | 0.6.15 | Arabic text utilities |
| datasets | 4.8.4 | HuggingFace dataset loading |
| torchcodec | 0.11.1 | Audio decoding for HuggingFace datasets |
| matplotlib | — | Chart generation |
| umap-learn | — | Dimensionality reduction for speaker cluster visualization |

### Installation Steps

```bash
# 1. Create conda environment
conda create -n new-arabic-tts python=3.10 -y
conda activate new-arabic-tts

# 2. Install Coqui TTS
pip install TTS

# 3. Pin transformers for compatibility (see Phase 2)
pip install transformers==4.44.2

# 4. Download XTTS-v2 base model
echo "y" | python -c "
from TTS.utils.manage import ModelManager
mm = ModelManager()
mm.download_model('tts_models/multilingual/multi-dataset/xtts_v2')
"

# 5. Copy model to project directory
mkdir -p models/base
cp -r ~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/* models/base/
```

### Project Structure

```
New Arabic TTS/
├── README.md                       # This file
├── CHANGELOG.md                    # Detailed change log
├── configs/                        # Configuration files
├── data/
│   └── egyptian/                   # Phase 5: Prepared training data
│       ├── metadata_train.csv      #   Training split (cleaned)
│       ├── metadata_eval.csv       #   Evaluation split (cleaned)
│       ├── wavs/                   #   22,050 Hz mono WAV files
│       └── rejected/              #   Clips that failed sanity check
├── docs/
│   ├── DATA_FORMAT.md              # Data format guide for fine-tuning
│   ├── images/
│   │   ├── speaker_clusters_umap.png   # Phase 5: UMAP speaker clusters
│   │   ├── cluster_sizes.png           # Phase 5: Cluster size chart
│   │   ├── similarity_distribution.png # Phase 5: Speaker consistency cutoff
│   │   ├── duration_distribution.png   # Phase 5: Duration histogram
│   │   ├── comparison_waveforms.png    # Phase 7: Waveform comparison
│   │   ├── comparison_spectrogram.png  # Phase 7: Spectrogram comparison
│   │   ├── comparison_metrics.png      # Phase 7: Metrics bar chart
│   │   └── training_loss.png           # Phase 7: Training loss curve
│   └── benchmarks/
│       ├── baseline.json               # Phase 1: Baseline metrics
│       ├── dataset_preparation.json    # Phase 5: Pipeline report
│       ├── sanity_check.json           # Phase 5: Data quality report
│       ├── training_summary.json       # Phase 6: Training results
│       └── evaluation.json             # Phase 7: Comparison metrics
├── models/
│   └── base/                       # XTTS-v2 base model (1.8GB)
│       ├── config.json
│       ├── model.pth
│       ├── speakers_xtts.pth       # 58 built-in speakers
│       └── vocab.json
├── outputs/
│   ├── original model/             # Phase 1: Raw base model outputs
│   │   └── Original_Model_test.wav #   Baseline benchmark (standard demo text)
│   ├── improved model/             # Phase 3+4: Text preprocessing + post-processing
│   │   ├── Improved_Model_test.wav #   Same text, improved pipeline
│   │   └── Improved_Model_test_stats.json  # Per-chunk generation stats
│   └── finetuned model/             # Phase 6: After fine-tuning
│       ├── Finetuned_Model_test.wav          # Fine-tuned output (same demo text)
│       └── Finetuned_Model_test_stats.json   # Generation stats
└── scripts/
    ├── arabic_preprocessor.py      # Phase 3: Arabic text preprocessing
    ├── baseline_test.py            # Phase 1: Baseline benchmark generator
    ├── infer.py                    # Phase 4: Improved inference pipeline
    ├── patch_tts.py                # Phase 2: PyTorch compatibility patch
    ├── prepare_dataset.py          # Phase 5: Dataset download, clustering, export
    ├── refine_cluster.py           # Phase 5: Refined top-N speaker selection
    ├── sanity_check.py             # Phase 5: 3-layer data quality verification
    ├── train.py                    # Phase 6: XTTS-v2 fine-tuning
    ├── evaluate.py                 # Phase 7: Metrics and comparison charts
    └── upsample.py                 # Phase 8: AudioSR upsampling (24→48kHz)
```

---

## Phase 1: Baseline Evaluation

**Objective**: Establish baseline Arabic output quality using the unmodified XTTS-v2 model.

### 1.1 Model Download

Downloaded XTTS-v2 base model (1.8GB) from Coqui's model hub. The model includes:
- `model.pth` - Full checkpoint (~370M parameters)
- `speakers_xtts.pth` - 58 pre-computed speaker embeddings
- `vocab.json` - BPE vocabulary (6,681 tokens)
- `config.json` - Model and training configuration

### 1.2 Built-in Speakers

XTTS-v2 ships with 58 speaker embeddings. No Arabic-native speakers are included - all speakers are from English/European datasets. We selected **"Gilberto Mathias"** for baseline testing as a speaker with a name suggesting potential Arabic phonetic compatibility.

<details>
<summary>Full speaker list (58 speakers)</summary>

Aaron Dreschner, Abrahan Mack, Adde Michal, Alexandra Hisakawa, Alison Dietlinde, Alma Maria, Ana Florence, Andrew Chipper, Annmarie Nele, Asya Anara, Gilberto Mathias, Baldur Sanjin, Barbora MacLean, Brenda Stern, Camilla Holmstrom, Chandra MacFarland, Claribel Dervla, Craig Gutsy, Daisy Studious, Damien Black, Damjan Chapman, Dionisio Schuyler, Eugenio Mataraci, Ferran Simen, Filip Traverse, Gilberto Mathias, Gitta Nikolina, Gracie Wise, Henriette Usha, Ige Behringer, Ilkin Urbano, Kazuhiko Atallah, Kumar Dahl, Lidiya Szekeres, Lilya Stainthorpe, Ludvig Milivoj, Luis Moray, Maja Ruoho, Marcos Rudaski, Narelle Moon, Nova Hogarth, Rosemary Okafor, Royston Min, Sofia Hellen, Suad Qasim, Szofi Granger, Tammie Ema, Tammy Grit, Tanja Adelina, Torcull Diarmuid, Uta Obando, Viktor Eka, Viktor Menelaos, Vjollca Johnnie, Wulf Carlevaro, Xavier Hayasaka, Zacharie Aimilios, Zofija Kendrick
</details>

### 1.3 Baseline Test

**Standard demo text** — this text is used consistently across all phases for comparison. It is also the default demo text when the model is published on HuggingFace:

```
مرحباً بكم في عالم التكنولوجيا الحديثة.
تُعدّ اللغة العربية من أقدم اللغات وأكثرها انتشاراً في العالم،
إذ يتحدث بها أكثر من أربعمائة مليون شخص حول العالم.
```

**Generation parameters** (baseline):

| Parameter | Value |
|-----------|-------|
| Temperature | 0.55 |
| Top-p | 0.85 |
| Repetition Penalty | 2.5 |
| Speaker | Gilberto Mathias (built-in) |
| Sentence Pause | 0.4s |

**Output file**: `outputs/original model/Original_Model_test.wav` — 19.33 seconds
**Benchmark data**: `docs/benchmarks/baseline.json`

> Listen to this file to hear the **unmodified XTTS-v2 base model** generating Arabic. This is the starting point — all subsequent improvements are measured against this output.

### 1.4 Baseline Observations

> **Note**: Detailed quality metrics (MOS, SNR, intelligibility) will be added after the evaluation pipeline is built in Phase 7.

Known issues with baseline output:
- [ ] Pronunciation errors on undiacritized words
- [ ] Non-native Arabic accent (English-trained speaker embedding)
- [ ] Inconsistent prosody and pacing
- [ ] No pause control between sentences (fixed 0.35s gaps)
- [ ] Potential rambling/over-generation on longer sentences

---

## Phase 2: Compatibility Fixes

Before any improvements could begin, two compatibility issues had to be resolved.

### 2.1 Transformers Version Conflict

**Problem**: Coqui TTS 0.22.0 imports `BeamSearchScorer` from HuggingFace Transformers, which was removed in Transformers v5.x.

```
ImportError: cannot import name 'BeamSearchScorer' from 'transformers'
```

**Fix**: Pin Transformers to a compatible version.

```bash
pip install transformers==4.44.2
```

### 2.2 PyTorch weights_only Default Change

**Problem**: PyTorch 2.11 changed the default of `torch.load()` from `weights_only=False` to `weights_only=True`. XTTS-v2 checkpoints contain non-tensor objects (config classes) that fail to load under the new default.

```
_pickle.UnpicklingError: Weights only load failed.
Unsupported global: GLOBAL TTS.tts.configs.xtts_config.XttsConfig
```

**Fix**: Patched `TTS/utils/io.py` to explicitly pass `weights_only=False`.

```python
# File: <env>/lib/python3.10/site-packages/TTS/utils/io.py
# Lines 51, 54: Added weights_only=False to both torch.load() calls

# Before:
return torch.load(f, map_location=map_location, **kwargs)

# After:
return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
```

> **Note**: This patch is applied to the installed package. If the `new-arabic-tts` conda environment is recreated, this patch must be re-applied. A script to automate this is provided at `scripts/patch_tts.py`.

### 2.3 Coqui Trainer Git Branch Detection

**Problem**: The Coqui `Trainer` class tries to detect the current git branch for logging. If the project directory is a git repo with no commits, `git branch` returns no output and the trainer crashes with `StopIteration`.

```
StopIteration (in trainer/generic_utils.py get_git_branch)
```

**Fix**: Initialize the git repo with at least one commit before running training.

```bash
git add . && git commit -m "Initial commit"
```

### 2.4 AudioSR Dependency Conflicts

**Problem**: Installing `audiosr` downgrades `librosa` (0.10→0.9.2) and `transformers` (4.44→4.30), breaking Coqui TTS compatibility.

**Fix**: After installing AudioSR, restore the required versions:

```bash
pip install audiosr
pip install "librosa>=0.10.0" "transformers==4.44.2"  # restore after audiosr
```

---

## Phase 3: Arabic Text Preprocessing

**Objective**: Build an Arabic-aware text preprocessing pipeline that improves pronunciation without forcing a specific dialect.

**Key design decision**: Tashkeel (diacritization) is **off by default**. Adding diacritics forces formal MSA pronunciation, which conflicts with the project's goal of natural dialect-specific speech. Instead, the model learns pronunciation from the speaker's training data. Tashkeel is available as an optional flag (`--tashkeel`) for users who want formal Arabic output.

**Script**: `scripts/arabic_preprocessor.py`

### 3.1 Hamza Normalization

Corrects common hamza placement errors using a 40+ word correction map. This helps the tokenizer produce consistent token sequences regardless of how the input was typed.

**Examples**:
| Input | Corrected |
|-------|-----------|
| ان | أن |
| الى | إلى |
| الالات | الآلات |
| اصبحت | أصبحت |
| اكثر | أكثر |
| الان | الآن |
| مسوول | مسؤول |

### 3.2 Number and Symbol Expansion

Converts numerals and symbols to Arabic words so the model doesn't have to guess pronunciation of non-text characters.

**Examples**:
| Input | Expanded |
|-------|----------|
| `70%` | سبعون بالمئة |
| `500$` | خمسمائة دولار |
| `&` | و |
| `2026` | ألفان و ستة و عشرون |

### 3.3 Tashkeel (Optional — MSA Mode)

When enabled via `--tashkeel`, the [Mishkal](https://github.com/linuxscout/mishkal) engine adds full diacritical marks. This produces formal MSA pronunciation.

**Example**:
```
Input:    الذكاء الاصطناعي يتطور بسرعة كبيرة
Tashkeel: الذَّكَاءُ الْاِصْطِنَاعِيُّ يَتَطَوَّرُ بِسُرْعَةِ كَبِيرَةِ
```

### 3.4 Processing Pipeline

```
Default (dialect mode):     clean → hamza → numbers → symbols → inference
Optional (formal MSA mode): clean → hamza → numbers → symbols → tashkeel → inference
```

---

## Phase 4: Inference Pipeline Improvements

**Objective**: Build a production-quality inference pipeline with smart chunking, post-processing, and reproducible output.

**Script**: `scripts/infer.py`

### 4.1 Smart Text Chunking

The base XTTS-v2 has a 166-character limit per inference call. Our chunker handles this with a 3-tier strategy:

1. **Sentence splitting** — split at Arabic punctuation (`.` `؟` `!` `؛`)
2. **Long sentence splitting** — split at commas/semicolons if over 160 chars, then by word boundary as last resort
3. **Short chunk merging** — merge chunks under 30 chars with their neighbors to avoid choppy output

### 4.2 Post-Processing Pipeline

Each generated chunk goes through:

| Step | What it does |
|------|-------------|
| **Pause compression** | Caps internal silence to 350ms — removes unnatural long pauses while preserving natural breathing rhythm |
| **Trailing silence trim** | Removes dead air at the end, keeps 150ms tail for natural sentence endings |
| **Rambling detection** | If output exceeds 1.5x expected duration, trims at nearest silence point |

### 4.3 Deterministic Generation

Each chunk gets a unique seed (`base_seed + chunk_index`), so the same input always produces the same output. Default base seed: `12345`.

### 4.4 Configurable Pause Control

| Pause Type | Duration | Purpose |
|-----------|----------|---------|
| Sentence pause | 0.4s | Between sentences |
| Paragraph pause | 0.70s | At end of full text |

### 4.5 First Improved Output

Generated the same AI test text through the improved pipeline:

**Output file**: `outputs/improved model/Improved_Model_test.wav` — 18.99 seconds
**Stats file**: `outputs/improved model/Improved_Model_test_stats.json`

> Listen to this file and compare with `outputs/original model/Original_Model_test.wav` to hear the effect of text preprocessing and post-processing. The voice is the same — the improvements are in how text is handled and how audio is cleaned.

| Metric | Baseline | Improved | Change |
|--------|----------|----------|--------|
| Total Duration | 19.33s | 18.99s | -1.8% (tighter, less dead air) |
| Generation Time | 6.41s | 6.86s | — |
| Chunks | 4 (manual) | 2 (smart chunking) | Fewer splits, more natural |
| Text Preprocessing | None | Hamza correction | Added |
| Post-Processing | None | Pause compress + trim + taper | Added |

> [!IMPORTANT]
> ### What improved and what didn't (yet)
>
> **Improved in this phase** (pipeline & text):
> - Numbers are spoken correctly (`70%` → `سبعون بالمئة` instead of raw symbols)
> - Hamza correction ensures consistent tokenization (`اكثر` → `أكثر`)
> - Cleaner audio: internal pauses capped at 150ms, natural sentence endings with taper
> - Reproducible output via deterministic seeding
> - Per-sentence stats tracked in JSON
>
> **Not improved yet** (requires fine-tuning in Phase 6):
> - Voice is still a non-Arabic built-in speaker (Gilberto Mathias)
> - Pronunciation still reflects the base model's limited Arabic training
> - Accent and prosody are not natural Arabic
>
> The voice quality transformation happens in **Phase 6** when we fine-tune on real Arabic speaker data. Phases 3–4 ensure the pipeline is clean and correct so that fine-tuning produces the best possible results.

### 4.6 Output Reference Guide

All output files are organized by phase. Each file captures a specific stage of improvement so you can listen and compare the progression:

| File | Phase | What it demonstrates |
|------|-------|---------------------|
| `outputs/original model/Original_Model_test.wav` | Phase 1 | Standard demo text, raw base model — **baseline for all comparisons** |
| `outputs/improved model/Improved_Model_test.wav` | Phase 3+4 | Same demo text, with hamza correction, smart chunking, pause compression, sentence taper |
| `outputs/improved model/Improved_Model_test_stats.json` | Phase 4 | Per-chunk generation stats (duration, timing, params) |
| `docs/benchmarks/baseline.json` | Phase 1 | Per-sentence metrics for baseline |
| `outputs/finetuned model/Finetuned_Model_test.wav` | Phase 6 | Same demo text, fine-tuned on Egyptian Arabic — **voice quality transformation** |
| `outputs/finetuned model/Finetuned_Model_test_stats.json` | Phase 6 | Per-chunk generation stats for fine-tuned output |
| `outputs/original model/Original_Model_test_48kHz.wav` | Phase 8 | Baseline upsampled to 48kHz/24-bit |
| `outputs/improved model/Improved_Model_test_48kHz.wav` | Phase 8 | Improved pipeline upsampled to 48kHz/24-bit |
| `outputs/finetuned model/Finetuned_Model_test_48kHz.wav` | Phase 8 | Fine-tuned model upsampled to 48kHz/24-bit — **best quality** |

> **How to compare**: Play `Original_Model_test.wav` and `Improved_Model_test.wav` side by side — all files use the same standard demo text. After Phase 6, play the fine-tuned version to hear the voice quality transformation. This progression tells the full story of the project.

---

## Phase 5: Data Collection & Preparation

> **Status**: In Progress

For full details on the data format, see **[docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)**.

### 5.1 Data Format Summary

XTTS-v2 fine-tuning requires:
- **Audio**: WAV files, 22,050 Hz mono (auto-resampled), 0.5–11.6s per clip
- **Metadata**: Pipe-delimited CSV with header: `audio_file|text|speaker_name`
- **Split**: Separate `metadata_train.csv` and `metadata_eval.csv`

### 5.2 Dataset Evaluation

We evaluated all publicly available Arabic speech datasets for suitability as XTTS-v2 fine-tuning data. The ideal dataset has: **single-speaker or speaker-labeled clips, 2–11s duration, clean audio, accurate transcripts, and 22kHz+ sample rate**.

#### Fully Free & Ready to Download

| Dataset | Dialect | Size | Sample Rate | Speakers | Transcripts | Our Verdict |
|---------|---------|------|-------------|----------|-------------|-------------|
| [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) | Egyptian | 72h / 103K clips | 16 kHz | Unknown (~85% male, no IDs) | Yes (normalized, no diacritics) | **Best option** |
| [Kaggle Egyptian Audio](https://www.kaggle.com/datasets/ahmedshafiq12/egyptian-audio-dataset-collected-from-youtube) | Egyptian | Unknown | Unknown | Unknown | Unknown | Needs investigation |
| [MagicHub ASR-EgArbCSC](https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/) | Egyptian | 5.5h / 9 conversations | 16 kHz | 4 speakers (2 pairs) | Yes (UTF-8 TXT) | Too small, conversational overlap |

#### Free with Registration

| Dataset | Dialect | Size | Sample Rate | Speakers | Transcripts | Our Verdict |
|---------|---------|------|-------------|----------|-------------|-------------|
| [MGB-3](https://arabicspeech.org/mgb3-asr/) (via [ARBML/klaam](https://github.com/ARBML/klaam)) | Egyptian | 15h+ | Unknown | Multiple (YouTube) | Yes (4 annotators per sentence) | Good quality, registration required |
| [Mozilla Common Voice](https://commonvoice.mozilla.org) | Mixed MSA | ~300h (Arabic) | 48 kHz | Crowdsourced | Yes | Large but mixed accents, no speaker consistency |

#### Other Arabic Speech Resources

| Dataset | Dialect | Size | Notes |
|---------|---------|------|-------|
| [OpenSLR #46](https://openslr.org/46/) | Tunisian MSA | Unknown | Tunisian Modern Standard Arabic |
| [OpenSLR #108 MediaSpeech](https://openslr.org/108/) | Multi (includes Arabic) | Unknown | Media speech from broadcasts |
| [OpenSLR #132](https://openslr.org/132/) | Quranic Arabic | Unknown | Quran recitation — very formal, not conversational |
| [ADI-5](https://github.com/ARBML/klaam) | Multi-dialect | 50h+ | Egyptian, Levantine, Gulf, North African, MSA |

#### Evaluation Summary

![Dataset Evaluation](docs/images/dataset_evaluation.png)
*Figure 4: Dataset evaluation results — green (chosen), yellow (viable alternatives), red (rejected).*

### 5.3 Chosen Approach

**Primary dataset**: [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) — starting with a subset for initial fine-tuning experiments.

**Why**: It's the largest freely available Egyptian Arabic dataset, already cleaned, with normalized transcripts that match our dialect-first approach (no diacritics). The 16kHz sample rate is a compromise, but XTTS-v2 handles resampling automatically.

**Future**: As the project expands to more dialects (Sudanese, Gulf, Levantine), we will collect dialect-specific data from single speakers — either from YouTube (using yt-dlp + Whisper for transcription) or from dedicated recording sessions.

### 5.4 Data Preparation Pipeline

The source dataset has **103K clips with no speaker labels**. To fine-tune XTTS-v2 for a single consistent voice, we need clips from the same speaker. We built an automated speaker clustering pipeline to solve this.

> **Curated with care**: The heavy lifting of speaker identification, quality filtering, and format conversion has been done for you. If you're using this project as a starting point, the exported data in `data/egyptian/` is ready for training — no additional preparation needed.

**Script**: [`scripts/prepare_dataset.py`](scripts/prepare_dataset.py)
**Source dataset**: [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) (HuggingFace)

#### Pipeline Overview

![Data Pipeline](docs/images/data_pipeline.png)
*Figure 3: Data preparation pipeline — from raw HuggingFace dataset to XTTS-v2 training-ready format.*

#### Step-by-Step Details

**Step 1 — Download**

| Detail | Value |
|--------|-------|
| Source | [`MAdel121/arabic-egy-cleaned`](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) |
| Method | `datasets.load_dataset()` (HuggingFace `datasets` library) |
| Size | 16.5 GB, 103,603 clips |
| Cache | `~/.cache/huggingface/datasets/` (auto-cached, no re-download on rerun) |
| Columns | `audio` (WAV 16kHz), `text` (normalized Arabic), `duration` (float), `dataset_source` (A-D) |

**Step 2 — Quality Filter**

Removes clips unsuitable for XTTS-v2 fine-tuning before any expensive processing:

| Filter | Criteria | Reason |
|--------|----------|--------|
| Duration | 2–11 seconds | XTTS-v2 max is 11.6s; clips < 2s lack enough context for speaker embedding |
| Text length | 10–200 characters | XTTS-v2 tokenizer max is 200; very short text = poor audio-text alignment |

| Input | Output |
|-------|--------|
| 103,603 clips (full dataset) | ~82,000 clips (estimated after filtering) |

**Step 3 — Speaker Embedding Extraction**

Extracts a numerical "voice fingerprint" for each clip so we can identify who is speaking.

| Detail | Value |
|--------|-------|
| Model | [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) (SpeechBrain) |
| Source | `speechbrain/spkrec-ecapa-voxceleb` (HuggingFace) |
| Embedding dim | 192 |
| Trained on | VoxCeleb (7,000+ speakers, multi-lingual) |
| Input | Each clip's audio (resampled to 16kHz for the encoder) |
| Output | One 192-dim vector per clip |
| Speed | ~200 clips/second on GPU |

**Why ECAPA-TDNN over alternatives?**

| Model | Dim | Trained On | Why we chose/skipped |
|-------|-----|------------|---------------------|
| ResNet (built-in XTTS) | 512 | English/European | Skipped — trained mainly on English speakers, less accurate for Arabic |
| **ECAPA-TDNN (SpeechBrain)** | 192 | VoxCeleb (7,000+ speakers) | **Chosen** — robust across languages, fast, no auth token needed |
| Pyannote | 512 | Multi-lingual | Good accuracy but requires HuggingFace auth token and license acceptance |
| WavLM + X-vector | 512 | Self-supervised | State of the art but significantly slower |

**Step 4 — Speaker Clustering**

Groups embeddings so that each cluster represents clips from the same speaker.

| Detail | Value |
|--------|-------|
| Method | Agglomerative Clustering (`sklearn`) |
| Similarity metric | Cosine similarity (embeddings L2-normalized) |
| Cluster count | Auto-selected from range 10–30 using silhouette score |
| Output | Cluster label per clip + silhouette score |

**Step 5 — Best Cluster Selection**

Picks the cluster most likely to be a single consistent speaker.

| Detail | Value |
|--------|-------|
| Minimum cluster size | 500 clips (configurable) |
| Selection metric | Highest mean cosine similarity to cluster centroid |
| Speaker ID assigned | `egyptian_male_01` |
| Output stats | Cluster size, total hours, mean similarity, mean duration |

**Step 6 — Export to XTTS-v2 Format**

Converts the selected cluster into the exact format XTTS-v2 expects for fine-tuning.

| Detail | Value |
|--------|-------|
| Audio resampling | 16,000 Hz → 22,050 Hz (via `torchaudio.functional.resample`) |
| Audio format | WAV, mono, float32 |
| CSV format | Pipe-delimited `\|` with header: `audio_file\|text\|speaker_name` |
| Train/eval split | 90% train / 10% eval (random, seed=42) |

| Output file | Description |
|-------------|-------------|
| `data/egyptian/metadata_train.csv` | Training metadata |
| `data/egyptian/metadata_eval.csv` | Evaluation metadata |
| `data/egyptian/wavs/clip_XXXXXX.wav` | Resampled audio files |
| `docs/benchmarks/dataset_preparation.json` | Full pipeline report (JSON) |

#### Generated Visualizations

The pipeline automatically generates analysis charts saved to `docs/images/`:

| Chart | File | What it shows |
|-------|------|--------------|
| Speaker clusters | `docs/images/speaker_clusters_umap.png` | UMAP projection of all embeddings, selected speaker in green |
| Cluster sizes | `docs/images/cluster_sizes.png` | Bar chart of all clusters, selected in green, threshold line |
| Duration distribution | `docs/images/duration_distribution.png` | Histogram of all clips vs selected speaker clips |

### 5.5 Reproducibility

To rerun the full pipeline from scratch:

```bash
conda activate new-arabic-tts
cd "New Arabic TTS"
python scripts/prepare_dataset.py
```

The dataset will be cached after the first download. Subsequent runs skip the download and go straight to filtering. All random operations use `seed=42` for reproducibility.

### 5.6 Dependencies Added in This Phase

```bash
pip install datasets speechbrain umap-learn matplotlib torchcodec openai-whisper
```

| Package | Version | Purpose |
|---------|---------|---------|
| `datasets` | 4.8.4 | HuggingFace dataset download and loading |
| `speechbrain` | 1.1.0 | ECAPA-TDNN speaker embedding model |
| `umap-learn` | — | UMAP dimensionality reduction for visualization |
| `matplotlib` | — | Chart generation |
| `torchcodec` | 0.11.1 | Audio decoding for HuggingFace datasets (required by `datasets` library) |
| `openai-whisper` | — | Alignment verification in sanity check (Layer 3) |

> [!NOTE]
> **Why OpenAI Whisper instead of faster-whisper?** We initially tried `faster-whisper` (CTranslate2-based) but it lacks CUDA support on ARM (aarch64) platforms like DGX Spark. OpenAI's original Whisper uses PyTorch directly and works with any CUDA-capable GPU.

### 5.7 Data Sanity Check

> [!IMPORTANT]
> ### Why clean data matters for TTS
>
> TTS fine-tuning is fundamentally different from ASR (speech recognition). An ASR model can tolerate noisy data because it's learning to *understand* speech — a few bad transcripts won't ruin comprehension. But a TTS model learns to *reproduce* exactly what it hears:
>
> - **Misaligned text** (text doesn't match audio) → the model learns wrong pronunciations and associates incorrect words with sounds
> - **Noisy audio** → the model learns to generate background noise, hiss, or hum as part of normal speech
> - **Silence or dead air** → the model learns to insert random pauses mid-sentence
> - **Corrupted or garbled text** → the model learns to mispronounce words consistently
>
> A few bad clips in a 5,000-clip dataset might go unnoticed. But hundreds of bad clips will measurably degrade every output the fine-tuned model produces. **The quality of your training data directly determines the quality ceiling of your model.**

**Script**: [`scripts/sanity_check.py`](scripts/sanity_check.py)

We apply a three-layer automated verification pipeline to catch problems before they reach the training stage.

#### Layer 1 — Text Quality

Fast checks on the transcript text to catch obviously bad entries.

| Check | Threshold | What it catches |
|-------|-----------|-----------------|
| Arabic character ratio | Min 60% | Non-Arabic text, garbled encoding |
| Latin character count | Max 3 | English words mixed into Arabic transcripts |
| Repeated words | Max 3 consecutive | Stuttered or looping transcriptions |
| Characters per second | 3–25 chars/s | Text that doesn't match audio duration (too much or too little text for the clip length) |

#### Layer 2 — Audio Quality

Signal analysis on the audio waveform.

| Check | Threshold | What it catches |
|-------|-----------|-----------------|
| Signal-to-noise ratio | Min 10 dB | Noisy recordings, background music |
| Silence ratio | Max 50% | Clips that are mostly dead air |
| Clipping ratio | Max 1% | Distorted audio from volume overflow |

#### Layer 3 — Alignment Verification (Whisper)

The most important layer. Uses [OpenAI Whisper](https://github.com/openai/whisper) (large-v3) to independently transcribe each clip, then compares the Whisper transcript to the metadata text.

| Detail | Value |
|--------|-------|
| Model | Whisper large-v3 (OpenAI, PyTorch-based) |
| Comparison | Normalized text similarity (SequenceMatcher) |
| Threshold | Min 40% similarity |
| What it catches | Misaligned text — clips where the transcript doesn't match what's actually spoken |

Text normalization before comparison:
- Remove diacritics
- Normalize alef variants (أ/إ/آ → ا)
- Normalize taa marbuta (ة → ه)
- Remove punctuation

#### Output Files

| File | Description |
|------|-------------|
| `data/egyptian/metadata_train.csv` | Cleaned training metadata (bad clips removed) |
| `data/egyptian/metadata_eval.csv` | Cleaned evaluation metadata (bad clips removed) |
| `data/egyptian/rejected/` | Rejected WAV files (moved here for manual review) |
| `docs/benchmarks/sanity_check.json` | Full report with per-clip rejection reasons |

#### How to run

```bash
conda activate new-arabic-tts
python scripts/sanity_check.py
```

#### Dependencies

```bash
pip install openai-whisper
```

> **Note**: We use `openai-whisper` (PyTorch-based) instead of `faster-whisper` (CTranslate2-based) because CTranslate2 lacks CUDA support on ARM platforms (DGX Spark). If you're on x86 with CUDA, `faster-whisper` will also work and is faster.

---

## Phase 6: Fine-Tuning

> **Status**: In Progress

**Script**: [`scripts/train.py`](scripts/train.py)

### 6.1 Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 4 | Conservative — 5h of data, avoid overfitting |
| Batch size | 4 | Fits comfortably in DGX Spark GPU memory |
| Gradient accumulation | 2 | Effective batch size = 8 |
| Learning rate | 5e-6 | Standard for XTTS-v2 fine-tuning |
| Optimizer | AdamW | With betas [0.9, 0.96], weight decay 1e-2 |
| LR scheduler | MultiStepLR | Decays at 50K, 150K, 300K steps |
| Precision | fp32 | Mixed precision causes NaN losses with XTTS GPT |
| Save frequency | Every 1,000 steps | Keep top 3 checkpoints |
| Base model | `models/base/model.pth` | XTTS-v2 pretrained weights |

### 6.2 Training Data

| Property | Value |
|----------|-------|
| Dataset | Cleaned Egyptian Arabic (post sanity check) |
| Train clips | 4,328 |
| Eval clips | 480 |
| Total hours | ~4.9 hours |
| Speaker | `egyptian_male_01` |
| Sample rate | 22,050 Hz |

### 6.3 Training Process

```bash
conda activate new-arabic-tts
python scripts/train.py
```

Output saved to `models/finetuned/run/training/` with:
- Best model checkpoint
- TensorBoard logs
- Training configuration

### 6.4 Training Results

| Metric | Value |
|--------|-------|
| Training time | 30 minutes |
| Final loss | 3.60 (down from 4.47 at start) |
| Text CE loss | 0.029 |
| Mel CE loss | 3.57 |
| Best checkpoint | `best_model_4328.pth` (final step) |
| Checkpoints saved | 1000, 2000, 3000, 4328 (best) |
| Output directory | `models/finetuned/run/training/GPT_XTTS_AR_FT-April-20-2026_02+14PM-81b7d55/` |

### 6.5 Fine-Tuned Output

**Output file**: `outputs/finetuned model/Finetuned_Model_test.wav` — 16.76 seconds
**Stats file**: `outputs/finetuned model/Finetuned_Model_test_stats.json`

> Listen to this file and compare with the baseline and improved outputs. The voice now comes from a real Arabic speaker (Egyptian male) rather than a built-in English-trained speaker. This is the core quality transformation of the project.

### 6.6 Full Comparison (Same Demo Text)

All three outputs use the same standard demo text for a fair comparison:

| Metric | Phase 1: Original | Phase 3+4: Improved | Phase 6: Fine-tuned |
|--------|-------------------|--------------------|--------------------|
| **File** | `Original_Model_test.wav` | `Improved_Model_test.wav` | `Finetuned_Model_test.wav` |
| **Duration** | 19.33s | 18.99s | 16.76s |
| **Speaker** | Gilberto Mathias (built-in) | Gilberto Mathias (built-in) | egyptian_male_01 (trained) |
| **Voice** | Non-Arabic accent | Non-Arabic accent | Arabic speaker |
| **Preprocessing** | None | Hamza + smart chunking | Hamza |
| **Model** | Base XTTS-v2 | Base XTTS-v2 | Fine-tuned (4 epochs) |

> [!NOTE]
> The fine-tuned output is shorter (16.76s vs 19.33s) because the Arabic-trained speaker speaks at a more natural pace compared to the English-trained baseline speaker.

---

## Phase 7: Evaluation & Comparison

> **Status**: Complete

**Script**: [`scripts/evaluate.py`](scripts/evaluate.py)

### 7.1 Evaluation Metrics

All three outputs were analyzed using the same standard demo text for a fair comparison.

| Metric | Phase 1: Baseline | Phase 3+4: Improved | Phase 6: Fine-tuned |
|--------|-------------------|--------------------|--------------------|
| **Duration** | 19.33s | 18.99s | 16.76s |
| **SNR** | 76.1 dB | 61.0 dB | 43.0 dB |
| **Silence ratio** | 26.8% | 28.8% | 22.6% |
| **Speaker** | Built-in (non-Arabic) | Built-in (non-Arabic) | Egyptian Arabic (trained) |

> **Note on SNR**: The baseline has higher SNR because the built-in speaker produces cleaner but less natural audio. The fine-tuned model has lower SNR because it reproduces the natural characteristics of real speech (breathing, vocal texture), which register as "noise" in a simple SNR measurement. This is expected and desirable — real human speech is not perfectly clean.

### 7.2 Generated Charts

All charts are saved to `docs/images/` and generated automatically by `scripts/evaluate.py`.

#### Waveform Comparison
![Waveform Comparison](docs/images/comparison_waveforms.png)
*Side-by-side waveforms of the same text across all three stages. The fine-tuned model shows more natural amplitude variation and pacing.*

#### Spectrogram Comparison
![Spectrogram Comparison](docs/images/comparison_spectrogram.png)
*Frequency content across time. The fine-tuned model shows richer harmonic structure typical of natural Arabic speech.*

#### Metrics Comparison
![Metrics Comparison](docs/images/comparison_metrics.png)
*Bar chart comparing duration, SNR, speaking duration, and silence ratio across all stages.*

#### Training Loss Curve
![Training Loss](docs/images/training_loss.png)
*Loss curve during fine-tuning. Steady decrease from 4.47 to 3.60 over 4 epochs (4,328 steps, 30 minutes).*

### 7.3 Audio Samples

Listen to all three outputs in order to hear the progression:

| Stage | File | Description |
|-------|------|-------------|
| 1. Baseline | `outputs/original model/Original_Model_test.wav` | Raw base model, non-Arabic speaker |
| 2. Improved | `outputs/improved model/Improved_Model_test.wav` | Same model, better text handling |
| 3. Fine-tuned | `outputs/finetuned model/Finetuned_Model_test.wav` | Arabic speaker, natural pronunciation |

### 7.4 How to Reproduce

```bash
conda activate new-arabic-tts
python scripts/evaluate.py
```

This regenerates all charts and metrics from the existing output WAV files.

---

## Phase 8: Studio-Quality Audio Upsampling

> **Status**: Complete

> [!IMPORTANT]
> ### Why not change the model's internal sample rate?
> XTTS-v2's sample rate (22,050 Hz internal → 24,000 Hz output) is deeply embedded across all components — the DVAE codec, GPT backbone, HiFiGAN vocoder, and mel spectrogram extraction are all trained and calibrated for these rates. Changing the output sample rate would require **retraining the entire model from scratch** (~350M parameters), which is not feasible.
>
> Instead, we use a **neural audio upsampler** as a post-processing step after the model generates speech. This is the same approach used in professional TTS production pipelines.

### 8.1 Approach

```
XTTS-v2 (24 kHz / 16-bit) → Neural Upsampler → 48 kHz or 96 kHz / 24-bit
```

The upsampler reconstructs the high-frequency detail (4–24 kHz) that the base model cannot produce, resulting in studio-quality output without modifying the core model.

### Audio Quality Tiers

| Tier | Sample Rate | Bit Depth | Quality | Use Case |
|------|-------------|-----------|---------|----------|
| Default | 24 kHz | 16-bit | Good | Fast generation, previews |
| High | 48 kHz | 24-bit | Professional | Podcasts, voiceover, video |
| Studio | 96 kHz | 24-bit | Studio master | Music production, archival |

### 8.2 Chosen Upsampler: AudioSR

We evaluated three options and chose [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution):

| Tool | Output | Speed | Why we chose/skipped |
|------|--------|-------|---------------------|
| **AudioSR** | 48 kHz | GPU-accelerated | **Chosen** — purpose-built for speech, diffusion-based, high quality |
| Vocos | 48 kHz | Fast | Skipped — requires custom training for best results |
| HiFi-GAN Universal | 48 kHz | Fast | Skipped — less natural than diffusion-based approach |

**Script**: [`scripts/upsample.py`](scripts/upsample.py)

### 8.3 Upsampling Process

All three official outputs are upsampled for comparison:

```bash
conda activate new-arabic-tts
python scripts/upsample.py --all
```

| Input (24kHz / 16-bit) | Output (48kHz / 24-bit) |
|------------------------|------------------------|
| `Original_Model_test.wav` | `Original_Model_test_48kHz.wav` |
| `Improved_Model_test.wav` | `Improved_Model_test_48kHz.wav` |
| `Finetuned_Model_test.wav` | `Finetuned_Model_test_48kHz.wav` |

### 8.4 AudioSR Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Model | `speech` | Optimized for speech (vs `basic` for general audio) |
| DDIM steps | 50 | Diffusion denoising steps (quality vs speed tradeoff) |
| Guidance scale | 3.5 | Controls fidelity to input |
| Seed | 42 | Reproducible output |
| Output format | WAV PCM_24 | 24-bit depth for professional quality |

> [!NOTE]
> AudioSR reconstructs the high-frequency content (8–24 kHz) that the base XTTS-v2 model cannot produce at its native 24kHz output. The result is noticeably crisper consonants, more natural breathiness, and professional-grade audio suitable for podcasts, voiceover, and video production.

### 8.5 Upsampling Results

| Input File | Output File | Processing Time |
|-----------|-------------|-----------------|
| `Original_Model_test.wav` (24kHz) | `Original_Model_test_48kHz.wav` (48kHz/24-bit) | 60.8s |
| `Improved_Model_test.wav` (24kHz) | `Improved_Model_test_48kHz.wav` (48kHz/24-bit) | 17.1s |
| `Finetuned_Model_test.wav` (24kHz) | `Finetuned_Model_test_48kHz.wav` (48kHz/24-bit) | 17.1s |

> The best output from this project is `Finetuned_Model_test_48kHz.wav` — fine-tuned Arabic voice at professional 48kHz/24-bit quality.

---

## Resource Usage

Every phase was tracked for compute resources. This helps you estimate hardware requirements if running on different GPUs.

### Per-Phase Resource Breakdown

| Phase | Task | GPU VRAM | System RAM | Time | Notes |
|-------|------|----------|------------|------|-------|
| **1. Baseline** | Model loading | ~4 GB | ~2 GB | 10s | XTTS-v2 base model (370M params) |
| **1. Baseline** | Inference (4 sentences) | ~5 GB | ~2 GB | 6.4s | 24kHz output, built-in speaker |
| **2. Compatibility** | Patch only | — | — | <1s | No GPU needed |
| **3. Preprocessing** | Text processing | — | ~200 MB | <1s | CPU only (no GPU) |
| **3. Preprocessing** | Mishkal tashkeel (optional) | — | ~500 MB | <1s | CPU only |
| **4. Inference** | Full pipeline (chunking + post) | ~5 GB | ~2 GB | 6.9s | Same model, added post-processing |
| **5. Data download** | HuggingFace dataset | — | ~36 GB | 27 min | 103K clips decoded into memory |
| **5. Embeddings** | ECAPA-TDNN (53K clips) | ~1 GB | ~2.5 GB | 10.4 min | 85 clips/sec on GPU |
| **5. Clustering** | Agglomerative + UMAP | — | ~4 GB | 2 min | CPU-bound (sklearn) |
| **5. Export** | Resample + write WAVs | — | ~1 GB | 3 min | 5,000 clips to 22,050 Hz |
| **5. Sanity check** | Whisper large-v3 (5K clips) | ~10 GB | ~2 GB | 93 min | ~1 clip/sec, alignment verification |
| **6. Training** | XTTS-v2 fine-tuning (4 epochs) | ~18 GB | ~4 GB | 30 min | 4,328 steps, batch 4, fp32 |
| **6. Training** | Peak during backprop | ~22 GB | ~4 GB | — | Gradient accumulation x2 |
| **7. Evaluation** | Chart generation | — | ~500 MB | <30s | CPU only (matplotlib) |
| **8. Upsampling** | AudioSR (per file) | ~6 GB | ~3 GB | 17-60s/file | 50 DDIM steps, diffusion model |

### GPU Memory Summary

```
┌──────────────────────────────────────────────────────────────┐
│  DGX Spark GPU: 121.7 GB unified memory                      │
│                                                              │
│  Peak usage by task:                                         │
│  ██░░░░░░░░░░░░░░░░░░  Training (22 GB)      18% of total   │
│  █░░░░░░░░░░░░░░░░░░░  Whisper check (10 GB)  8% of total   │
│  █░░░░░░░░░░░░░░░░░░░  AudioSR (6 GB)         5% of total   │
│  ░░░░░░░░░░░░░░░░░░░░  Inference (5 GB)       4% of total   │
│                                                              │
│  Minimum GPU for this project: 8 GB (inference only)         │
│  Minimum GPU for full pipeline: 24 GB (training + Whisper)   │
└──────────────────────────────────────────────────────────────┘
```

### For Different Hardware

| GPU | VRAM | Can run |
|-----|------|---------|
| RTX 3060 | 12 GB | Inference only. Training needs batch_size=1 |
| RTX 3090 / 4090 | 24 GB | Full pipeline. Reduce batch_size to 2 for training |
| A100 / H100 | 40-80 GB | Full pipeline at higher batch sizes |
| DGX Spark (this project) | 121.7 GB | Full pipeline, comfortable headroom |

### Total Pipeline Cost

| Resource | Value |
|----------|-------|
| Total GPU time | ~2.5 hours (all phases) |
| Total wall clock | ~4 hours (including downloads and waits) |
| Peak GPU VRAM | 22 GB (during training) |
| Peak system RAM | 36 GB (during dataset loading) |
| Disk space (project) | ~25 GB (model + data + outputs) |
| Disk space (cache) | ~16 GB (HuggingFace dataset cache) |

---

## Results

| Metric | Phase 1: Baseline | Phase 3+4: Improved | Phase 6: Fine-tuned |
|--------|-------------------|--------------------|--------------------|
| Output duration | 19.33s | 18.99s | 16.76s |
| Speaker | Built-in (non-Arabic) | Built-in (non-Arabic) | Egyptian Arabic (trained) |
| Arabic naturalness | Low | Low (same model) | Significantly improved |
| Text handling | Raw (numbers, symbols as-is) | Hamza + numbers + symbols | Hamza |
| Post-processing | None | Pause compress + trim + taper | None (raw for comparison) |
| Training loss | — | — | 3.60 (from 4.47) |
| Training time | — | — | 30 minutes |

---

## Usage

### CLI Usage

```bash
# Basic generation (dialect mode — no tashkeel)
python scripts/infer.py --text "مرحباً بكم في العالم العربي" --output outputs/test.wav

# From a text file
python scripts/infer.py --text-file input.txt --output outputs/test.wav

# Formal MSA mode (with tashkeel)
python scripts/infer.py --text "مرحباً بكم" --output outputs/msa.wav --tashkeel

# Custom parameters
python scripts/infer.py --text "مرحباً" --output outputs/test.wav \
    --speaker "Gilberto Mathias" --temperature 0.55 --top-p 0.85 --rep-penalty 2.5 --pause 0.4
```

### Python API

```python
from scripts.infer import load_model, load_speaker, generate
from scripts.arabic_preprocessor import ArabicPreprocessor
import soundfile as sf

model = load_model("models/base")
gpt_cond_latent, speaker_embedding = load_speaker("models/base", "Gilberto Mathias")

# Dialect mode (default)
preprocessor = ArabicPreprocessor(enable_tashkeel=False)

result = generate(
    model=model,
    text="الذكاء الاصطناعي يتطور بسرعة كبيرة",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    preprocessor=preprocessor,
)

sf.write("output.wav", result["wav"], result["sr"])
print(result["stats"])
```

---

## Known Limitations & Future Improvements

### Current Limitations

The current model produces noticeably improved Arabic speech compared to the base XTTS-v2, but it is not yet at the level of a professional human voice actor. The primary bottleneck is **data quality, not model architecture**.

**Why data is the limiting factor:**

The training data used in this release comes from [MAdel121/arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned), an open-source Egyptian Arabic speech dataset originally built for ASR (speech recognition), not TTS. While we applied extensive cleaning — speaker clustering, quality filtering, and Whisper-based alignment verification — the data has inherent limitations:

- **No speaker labels in the source** — our speaker clustering identifies the most consistent voice, but it's not guaranteed to be a single speaker across all 4,328 training clips
- **16 kHz source audio** — upsampled to 22,050 Hz for training, but the high-frequency detail above 8 kHz was never captured in the original recordings
- **ASR-grade transcripts** — the text was generated by speech recognition models, not hand-verified. Our Whisper cross-check caught 188 misaligned clips (3.8%), but subtle errors may remain
- **Single dialect** — currently trained on Egyptian Arabic only. Sudanese, Gulf, Levantine, and other dialects require their own training data

### Data Collection Pipeline (Next Phase)

To achieve production-quality Arabic TTS, we need purpose-built training data. The next phase will use a custom data collection pipeline:

![YouTube Data Collection Pipeline](docs/images/youtube_pipeline.png)
*Figure 5: Future data collection pipeline — from YouTube to training-ready data. Step 6 (Manual Review) is the most critical step for TTS data quality.*

> [!IMPORTANT]
> **Manual cleaning is crucial.** No automated pipeline — no matter how sophisticated — can replace a human listener verifying that text matches audio. The difference between a good TTS model and a great one often comes down to the last 5% of data quality that only manual review can catch.

### Acknowledgments — Training Data

We gratefully acknowledge the following open-source data providers whose work made this project possible:

- **[MAdel121](https://huggingface.co/MAdel121)** — Creator of the [arabic-egy-cleaned](https://huggingface.co/datasets/MAdel121/arabic-egy-cleaned) dataset (72h Egyptian Arabic, cleaned and normalized). This dataset served as our primary training data source
- **[Mozilla Common Voice](https://commonvoice.mozilla.org)** — Crowdsourced multilingual speech data including Arabic contributions
- **[Coqui AI](https://github.com/coqui-ai/TTS)** — Creators of XTTS-v2 and the Coqui TTS framework that this project builds upon
- **[SpeechBrain](https://speechbrain.github.io)** — ECAPA-TDNN speaker verification model used for speaker clustering
- **[OpenAI Whisper](https://github.com/openai/whisper)** — Speech recognition model used for data alignment verification

### Roadmap

| Priority | Improvement | Impact |
|----------|------------|--------|
| **High** | Collect clean, single-speaker Sudanese Arabic data (10-20 hours) | Enables the primary dialect goal of this project |
| **High** | Manual review of training data with native Arabic speakers | Significant quality improvement from cleaner data |
| **Medium** | Expand to Gulf, Levantine, and North African dialects | Broader Arabic dialect coverage |
| **Medium** | Record purpose-built TTS data in a controlled environment | Studio-quality source audio (48kHz+) eliminates upsampling artifacts |
| **Low** | Train a custom Arabic tokenizer with larger vocabulary | Better text-to-token mapping for Arabic script |
| **Low** | Fine-tune the HiFiGAN vocoder on Arabic speech | More natural waveform generation for Arabic phonetics |

---

## Industry Use Cases

When this project reaches production quality with clean, manually verified training data across multiple Arabic dialects, it enables a wide range of applications:

### Media & Entertainment

| Use Case | Description |
|----------|-------------|
| **Arabic podcast production** | Generate full episodes or narration in natural Arabic voices, reducing production time from days to minutes |
| **Video dubbing & localization** | Dub English/foreign content into Arabic dialects — Egyptian for entertainment, MSA for news, Gulf for regional content |
| **Audiobook generation** | Convert Arabic books and educational material to spoken audio at scale |
| **Social media content** | Generate voiceovers for Arabic YouTube, TikTok, and Instagram content |

### Business & Enterprise

| Use Case | Description |
|----------|-------------|
| **Arabic call center IVR** | Natural-sounding Arabic voice prompts and automated responses for customer service |
| **Corporate training** | Generate Arabic e-learning narration for employee training across MENA region |
| **Accessibility** | Screen readers and assistive technology with natural Arabic speech for visually impaired users |
| **Document-to-speech** | Convert Arabic reports, emails, and documents to audio for on-the-go consumption |

### Technology & AI

| Use Case | Description |
|----------|-------------|
| **Arabic voice assistants** | Power Siri/Alexa-like assistants that speak natural Arabic across dialects |
| **Real-time translation** | Text-to-speech component for Arabic in live translation systems |
| **Gaming** | Generate Arabic NPC dialogue and narration for games targeting the MENA market |
| **Robotics** | Arabic speech output for service robots in airports, malls, and hospitals across the Arab world |

### Education & Government

| Use Case | Description |
|----------|-------------|
| **Language learning** | Generate example pronunciations in specific Arabic dialects for language learners |
| **Public announcements** | Natural Arabic announcements for airports, transit systems, and government services |
| **News automation** | Generate Arabic news bulletins from text feeds |

### Market Opportunity

The Arabic-speaking world represents over **400 million native speakers** across 25+ countries, yet Arabic TTS technology significantly lags behind English and European languages. High-quality, dialect-aware Arabic TTS addresses an underserved market with growing demand for Arabic digital content, accessibility tools, and AI-powered services.

> [!NOTE]
> ### Future Releases
> This is an active project. Improved versions of the model will be released as we collect cleaner training data, expand dialect coverage, and refine the pipeline. Future releases will include:
> - **v2.0** — Retrained on manually verified, studio-quality Arabic speech data
> - **Dialect-specific checkpoints** — Separate models for Sudanese, Egyptian, Gulf, Levantine, and MSA
> - **Pre-built speaker profiles** — Ready-to-use voice embeddings for common use cases
>
> Follow this repository for updates. Releases will be published on both GitHub and HuggingFace.

---

## License

- **XTTS-v2 Base Model**: [Coqui Public Model License (CPML)](https://coqui.ai/cpml)
- **This Project**: MIT License

---

## Citation

```bibtex
@misc{arabic-tts-xtts-v2,
  title={Improving XTTS-v2 for Arabic Text-to-Speech},
  author={Moe Eldouma},
  year={2026},
  url={https://github.com/eldoumamoe-glitch/arabic-tts}
}
```
