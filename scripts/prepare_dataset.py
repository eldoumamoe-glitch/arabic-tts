"""
Dataset preparation pipeline for Arabic TTS fine-tuning.

Downloads the MAdel121/arabic-egy-cleaned dataset from HuggingFace,
extracts speaker embeddings using ECAPA-TDNN, clusters them to find
consistent single-speaker groups, and exports the best cluster in
XTTS-v2 training format.

Speaker clustering approach:
  The source dataset has 103K clips with NO speaker labels (~85% male,
  mixed speakers). To fine-tune XTTS-v2 for a consistent voice, we need
  clips from the same speaker. This script uses ECAPA-TDNN (SpeechBrain)
  to extract speaker embeddings, then clusters them with Agglomerative
  Clustering to identify speaker groups. The largest high-quality cluster
  is selected and exported.

Usage:
    conda activate new-arabic-tts
    python scripts/prepare_dataset.py

Output:
    data/egyptian/
    ├── metadata_train.csv
    ├── metadata_eval.csv
    ├── wavs/
    └── speaker_analysis.json

    docs/images/
    ├── speaker_clusters_umap.png
    ├── cluster_sizes.png
    └── duration_distribution.png
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "egyptian"
WAVS_DIR = DATA_DIR / "wavs"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

# --- Configuration ---
MIN_DURATION = 2.0      # seconds — minimum clip duration for training
MAX_DURATION = 11.0     # seconds — XTTS-v2 max
MIN_TEXT_LEN = 10       # characters
MAX_TEXT_LEN = 200      # XTTS-v2 max
TARGET_SR = 22050       # XTTS-v2 internal sample rate
EVAL_SPLIT = 0.1        # 10% for evaluation
RANDOM_SEED = 42
SPEAKER_NAME = "egyptian_male_01"

# Clustering
N_CLUSTERS_RANGE = (10, 30)  # search range for optimal cluster count
MIN_CLUSTER_SIZE = 500       # minimum clips per usable cluster


def download_dataset():
    """Download the HuggingFace dataset."""
    from datasets import load_dataset
    print("[1/6] Downloading dataset from HuggingFace...")
    print("      This may take 10-30 minutes depending on your connection.")
    t0 = time.time()
    ds = load_dataset("MAdel121/arabic-egy-cleaned", split="train")
    elapsed = time.time() - t0
    print(f"      Downloaded {len(ds)} clips in {elapsed/60:.1f} minutes")
    return ds


def extract_embeddings(ds):
    """Extract ECAPA-TDNN speaker embeddings for all clips."""
    from speechbrain.inference.speaker import EncoderClassifier

    print("[2/6] Extracting speaker embeddings (ECAPA-TDNN)...")
    print(f"      Processing {len(ds)} clips...")

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    embeddings = []
    valid_indices = []
    durations = []
    skipped = 0
    t0 = time.time()

    for i in range(len(ds)):
        sample = ds[i]
        audio = sample["audio"]
        duration = sample.get("duration", len(audio["array"]) / audio["sampling_rate"])
        text = sample.get("text", "")

        # Filter by duration and text length
        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped += 1
            continue
        if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
            skipped += 1
            continue

        # Convert to tensor
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]

        # Resample to 16kHz for ECAPA-TDNN (expects 16kHz)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Extract embedding
        with torch.no_grad():
            emb = classifier.encode_batch(waveform.cuda())
            embeddings.append(emb.squeeze().cpu().numpy())

        valid_indices.append(i)
        durations.append(duration)

        if (i + 1) % 1000 == 0:
            rate = (len(valid_indices)) / (time.time() - t0)
            eta = (len(ds) - i) / max(rate, 1) / 60
            print(f"      [{i+1:,}/{len(ds):,}] {len(valid_indices):,} valid, "
                  f"{skipped:,} skipped, {rate:.0f} clips/s, ETA {eta:.0f}min")

    elapsed = time.time() - t0
    embeddings = np.array(embeddings)
    print(f"      Done: {len(embeddings):,} valid clips, {skipped:,} skipped, "
          f"{elapsed/60:.1f} minutes")

    return embeddings, valid_indices, durations


def cluster_speakers(embeddings):
    """Cluster speaker embeddings to find consistent speaker groups."""
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    print("[3/6] Clustering speakers...")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    # Find optimal number of clusters using silhouette score
    best_score = -1
    best_n = 15
    best_labels = None

    for n in range(N_CLUSTERS_RANGE[0], N_CLUSTERS_RANGE[1] + 1, 5):
        clustering = AgglomerativeClustering(n_clusters=n)
        labels = clustering.fit_predict(embeddings_norm)
        score = silhouette_score(embeddings_norm, labels, sample_size=min(5000, len(labels)))
        print(f"      n_clusters={n:2d} → silhouette={score:.3f}")
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels

    print(f"      Best: {best_n} clusters (silhouette={best_score:.3f})")

    # Analyze clusters
    cluster_info = []
    for c in range(best_n):
        mask = best_labels == c
        cluster_info.append({
            "cluster_id": int(c),
            "size": int(mask.sum()),
            "percentage": round(float(mask.sum()) / len(best_labels) * 100, 1),
        })

    cluster_info.sort(key=lambda x: x["size"], reverse=True)
    print("\n      Top 5 clusters:")
    for info in cluster_info[:5]:
        bar = "█" * (info["size"] // 100)
        print(f"      Cluster {info['cluster_id']:2d}: {info['size']:5,} clips "
              f"({info['percentage']:5.1f}%) {bar}")

    return best_labels, best_n, best_score, cluster_info


def select_best_cluster(labels, cluster_info, embeddings, durations):
    """Select the best cluster based on size and embedding consistency."""
    print("[4/6] Selecting best speaker cluster...")

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    # Score each large cluster by internal consistency (cosine similarity)
    candidates = [c for c in cluster_info if c["size"] >= MIN_CLUSTER_SIZE]
    if not candidates:
        # Fallback: use the largest cluster
        candidates = [cluster_info[0]]
        print(f"      Warning: no cluster >= {MIN_CLUSTER_SIZE} clips, "
              f"using largest ({candidates[0]['size']} clips)")

    for c in candidates:
        mask = labels == c["cluster_id"]
        cluster_embs = embeddings_norm[mask]
        # Mean cosine similarity to centroid
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        similarities = cluster_embs @ centroid
        c["mean_similarity"] = round(float(similarities.mean()), 4)
        c["std_similarity"] = round(float(similarities.std()), 4)
        # Duration stats
        cluster_durations = np.array(durations)[mask]
        c["mean_duration"] = round(float(cluster_durations.mean()), 2)
        c["total_hours"] = round(float(cluster_durations.sum()) / 3600, 2)

    # Pick cluster with best consistency among large enough clusters
    candidates.sort(key=lambda x: x["mean_similarity"], reverse=True)
    best = candidates[0]

    print(f"      Selected Cluster {best['cluster_id']}:")
    print(f"        Clips:          {best['size']:,}")
    print(f"        Total hours:    {best['total_hours']}")
    print(f"        Mean similarity: {best['mean_similarity']}")
    print(f"        Mean duration:  {best['mean_duration']}s")

    return best


def generate_visuals(embeddings, labels, cluster_info, best_cluster, durations):
    """Generate analysis charts."""
    print("[5/6] Generating visualizations...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    # --- 1. UMAP Speaker Clusters ---
    try:
        from umap import UMAP
        print("      Computing UMAP projection (this may take a minute)...")
        # Subsample for UMAP if too many points
        max_points = 10000
        if len(embeddings_norm) > max_points:
            idx = np.random.RandomState(RANDOM_SEED).choice(
                len(embeddings_norm), max_points, replace=False
            )
            emb_sub = embeddings_norm[idx]
            labels_sub = labels[idx]
        else:
            emb_sub = embeddings_norm
            labels_sub = labels
            idx = np.arange(len(embeddings_norm))

        reducer = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30)
        projected = reducer.fit_transform(emb_sub)

        fig, ax = plt.subplots(figsize=(12, 8))
        # Plot non-selected clusters in gray
        best_id = best_cluster["cluster_id"]
        other_mask = labels_sub != best_id
        selected_mask = labels_sub == best_id

        ax.scatter(
            projected[other_mask, 0], projected[other_mask, 1],
            c="lightgray", s=3, alpha=0.3, label="Other speakers"
        )
        ax.scatter(
            projected[selected_mask, 0], projected[selected_mask, 1],
            c="#2ecc71", s=8, alpha=0.6, label=f"Selected: {SPEAKER_NAME}"
        )
        ax.set_title("Speaker Embedding Clusters (UMAP Projection)", fontsize=14)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(fontsize=11, markerscale=3)
        ax.set_facecolor("#fafafa")
        fig.tight_layout()
        fig.savefig(IMAGES_DIR / "speaker_clusters_umap.png", dpi=150)
        plt.close(fig)
        print("      Saved speaker_clusters_umap.png")
    except Exception as e:
        print(f"      UMAP failed: {e}")

    # --- 2. Cluster Size Distribution ---
    fig, ax = plt.subplots(figsize=(10, 5))
    sizes = [c["size"] for c in cluster_info]
    cluster_ids = [f"C{c['cluster_id']}" for c in cluster_info]
    colors = ["#2ecc71" if c["cluster_id"] == best_cluster["cluster_id"]
              else "#95a5a6" for c in cluster_info]

    ax.bar(range(len(sizes)), sizes, color=colors)
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels(cluster_ids, rotation=45, fontsize=8)
    ax.set_ylabel("Number of Clips")
    ax.set_title("Speaker Cluster Sizes (green = selected for training)")
    ax.axhline(y=MIN_CLUSTER_SIZE, color="red", linestyle="--",
               alpha=0.5, label=f"Min threshold ({MIN_CLUSTER_SIZE})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cluster_sizes.png", dpi=150)
    plt.close(fig)
    print("      Saved cluster_sizes.png")

    # --- 3. Duration Distribution ---
    best_mask = labels == best_cluster["cluster_id"]
    selected_durations = np.array(durations)[best_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(durations, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].set_title("All Valid Clips")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Count")

    axes[1].hist(selected_durations, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"Selected Speaker ({best_cluster['size']:,} clips)")
    axes[1].set_xlabel("Duration (seconds)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Audio Duration Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "duration_distribution.png", dpi=150)
    plt.close(fig)
    print("      Saved duration_distribution.png")


def export_dataset(ds, valid_indices, labels, best_cluster, durations):
    """Export selected cluster to XTTS-v2 training format."""
    print("[6/6] Exporting dataset to XTTS-v2 format...")

    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    best_id = best_cluster["cluster_id"]

    # Get indices of clips in the best cluster
    selected = []
    for i, (ds_idx, label, dur) in enumerate(zip(valid_indices, labels, durations)):
        if label == best_id:
            selected.append((ds_idx, dur))

    print(f"      Exporting {len(selected):,} clips...")

    rows = []
    t0 = time.time()
    for j, (ds_idx, dur) in enumerate(selected):
        sample = ds[ds_idx]
        audio = sample["audio"]
        text = sample["text"].strip()

        # Convert to target sample rate
        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        # Save WAV
        wav_filename = f"clip_{j:06d}.wav"
        wav_path = WAVS_DIR / wav_filename
        sf.write(str(wav_path), waveform.squeeze().numpy(), TARGET_SR)

        rows.append(f"wavs/{wav_filename}|{text}|{SPEAKER_NAME}")

        if (j + 1) % 500 == 0:
            print(f"      [{j+1:,}/{len(selected):,}] exported...")

    # Shuffle and split
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(rows)
    split_idx = int(len(rows) * (1 - EVAL_SPLIT))

    header = "audio_file|text|speaker_name"

    train_path = DATA_DIR / "metadata_train.csv"
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows[:split_idx]) + "\n")

    eval_path = DATA_DIR / "metadata_eval.csv"
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("\n".join(rows[split_idx:]) + "\n")

    elapsed = time.time() - t0
    print(f"      Exported in {elapsed/60:.1f} minutes")
    print(f"      Train: {split_idx:,} clips → {train_path}")
    print(f"      Eval:  {len(rows) - split_idx:,} clips → {eval_path}")

    return split_idx, len(rows) - split_idx


def main():
    print("=" * 70)
    print("  Arabic TTS Dataset Preparation Pipeline")
    print("  Speaker clustering & export for XTTS-v2 fine-tuning")
    print("=" * 70)
    print()

    t_start = time.time()

    # Step 1: Download
    ds = download_dataset()

    # Step 2: Extract embeddings
    embeddings, valid_indices, durations = extract_embeddings(ds)

    # Step 3: Cluster
    labels, n_clusters, silhouette, cluster_info = cluster_speakers(embeddings)

    # Step 4: Select best cluster
    best_cluster = select_best_cluster(labels, cluster_info, embeddings, durations)

    # Step 5: Generate visuals
    generate_visuals(embeddings, labels, cluster_info, best_cluster, durations)

    # Step 6: Export
    n_train, n_eval = export_dataset(ds, valid_indices, labels, best_cluster, durations)

    # Save analysis report
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": "MAdel121/arabic-egy-cleaned",
        "total_source_clips": len(ds),
        "valid_after_filtering": len(valid_indices),
        "filtering_criteria": {
            "min_duration_s": MIN_DURATION,
            "max_duration_s": MAX_DURATION,
            "min_text_length": MIN_TEXT_LEN,
            "max_text_length": MAX_TEXT_LEN,
        },
        "clustering": {
            "method": "ECAPA-TDNN (SpeechBrain) + Agglomerative Clustering",
            "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
            "embedding_dim": 192,
            "n_clusters": n_clusters,
            "silhouette_score": round(silhouette, 4),
        },
        "selected_cluster": best_cluster,
        "speaker_name": SPEAKER_NAME,
        "export": {
            "train_clips": n_train,
            "eval_clips": n_eval,
            "target_sample_rate": TARGET_SR,
            "eval_split": EVAL_SPLIT,
            "format": "XTTS-v2 (pipe-delimited CSV + WAV)",
        },
        "total_pipeline_time_min": round((time.time() - t_start) / 60, 1),
    }

    report_path = BENCHMARKS_DIR / "dataset_preparation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("  Pipeline Complete!")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} minutes")
    print()
    print(f"  Speaker: {SPEAKER_NAME}")
    print(f"  Train:   {n_train:,} clips")
    print(f"  Eval:    {n_eval:,} clips")
    print(f"  Hours:   {best_cluster['total_hours']}")
    print()
    print(f"  Data:    {DATA_DIR}")
    print(f"  Report:  {report_path}")
    print(f"  Charts:  {IMAGES_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
