# Changelog

All notable changes to this project are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] - 2026-04-20

### Added
- **Refined speaker clustering** (`scripts/refine_cluster.py`)
  - Takes top N most consistent clips from the largest cluster by cosine similarity to centroid
  - Result: 5,000 clips (5.08 hours) with 0.8139 mean similarity
  - Generates similarity distribution chart (`docs/images/similarity_distribution.png`)
- **Data sanity check pipeline** (`scripts/sanity_check.py`)
  - Layer 1 — Text quality: Arabic ratio, Latin chars, repeated words, chars-per-second rate
  - Layer 2 — Audio quality: SNR, silence ratio, clipping detection
  - Layer 3 — Alignment verification: Whisper large-v3 cross-check against metadata text
  - Rejected clips moved to `data/egyptian/rejected/` for manual review
  - Full report saved to `docs/benchmarks/sanity_check.json`
- **Dependencies**: Added `faster-whisper` for alignment verification

### Changed
- Speaker changed from Badr Odhiambo to **Gilberto Mathias** (deeper male voice)
- Standard demo text updated to simpler Arabic
- Output files renamed: `Original_Model_test.wav`, `Improved_Model_test.wav`
- Dataset strategy changed from "smallest most consistent cluster" (556 clips) to "top 5,000 from largest cluster" (5.08 hours) for better fine-tuning results

## [0.3.0] - 2026-04-20

### Added
- **Dataset preparation pipeline** (`scripts/prepare_dataset.py`)
  - Downloads MAdel121/arabic-egy-cleaned from HuggingFace (103K clips, 72h Egyptian Arabic)
  - Quality filtering: 2–11s duration, 10–200 char text
  - Speaker embedding extraction using ECAPA-TDNN (SpeechBrain, 192-dim, trained on 7000+ speakers)
  - Agglomerative clustering with silhouette scoring to find same-speaker groups
  - Automatic best-cluster selection by cosine similarity consistency
  - Export to XTTS-v2 format: 22,050 Hz WAV + pipe-delimited CSV, 90/10 train/eval split
  - Generated speaker ID: `egyptian_male_01`
- **Visualization charts** (`docs/images/`)
  - `speaker_clusters_umap.png` — UMAP projection of speaker embeddings
  - `cluster_sizes.png` — cluster size distribution
  - `duration_distribution.png` — audio duration histogram
- **Data format documentation** (`docs/DATA_FORMAT.md`)
  - Complete guide for XTTS-v2 training data format
  - HuggingFace dataset conversion example
  - Quality checklist and Arabic data sources table
- **Dependencies**: Added `datasets`, `speechbrain`, `umap-learn`, `matplotlib`

## [0.2.0] - 2026-04-20

### Added
- **Arabic text preprocessor** (`scripts/arabic_preprocessor.py`)
  - Hamza normalization: 40+ word correction map (ان→أن, الى→إلى, الالات→الآلات, etc.)
  - Number-to-word expansion: Arabic numerals via `num2words` (70% → سبعون بالمئة)
  - Symbol expansion: &→و, $→دولار, %→بالمئة, etc.
  - Optional tashkeel (Mishkal engine) for formal MSA mode — **off by default** to preserve natural dialect pronunciation
- **Improved inference pipeline** (`scripts/infer.py`)
  - Smart 3-tier text chunking (sentence → comma → word boundary)
  - Short chunk merging (min 30 chars)
  - Post-processing: pause compression (150ms cap), trailing silence trim, sentence taper (400ms quadratic fade), rambling detection
  - Deterministic seeding per chunk for reproducible output
  - Configurable sentence/paragraph pauses (0.35s / 0.70s)
  - JSON stats output alongside each WAV
  - CLI with full parameter control
- **Dependencies**: Added `mishkal` and `pyarabic` for Arabic NLP

### Output
- `outputs/improved model/AIImproved.wav` — 53.72s (same test text as baseline)

## [0.1.0] - 2026-04-20

### Added
- Project initialized with XTTS-v2 base model (Coqui TTS 0.22.0)
- Fresh conda environment `new-arabic-tts` (Python 3.10, PyTorch 2.11.0+cu130)
- Base model downloaded and stored at `models/base/` (1.8GB checkpoint, 58 speakers)
- Baseline Arabic output generated using built-in "Badr Odhiambo" speaker
- Output: `outputs/original model/test_output.wav` (single sentence, 4.87s)
- Output: `outputs/original model/AIOriginal.wav` (10 sentences on AI, 54.70s)
- Project documentation structure (README.md, CHANGELOG.md, docs/)

### Fixed
- **Transformers compatibility**: Pinned `transformers==4.44.2` to resolve `BeamSearchScorer` import error in TTS 0.22.0 (removed in Transformers 5.x)
- **PyTorch weights_only**: Patched `TTS/utils/io.py` to pass `weights_only=False` to `torch.load()` — required for PyTorch 2.11+ which changed the default from `False` to `True`
