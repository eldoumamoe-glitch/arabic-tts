"""
Refine speaker cluster: take the largest cluster and extract the most
consistent subset by cosine similarity to centroid.

This script reuses the embeddings already extracted by prepare_dataset.py
but applies a smarter selection strategy:
  1. Load all embeddings from the previous run
  2. Take the largest cluster
  3. Rank clips by cosine similarity to cluster centroid
  4. Keep top N most consistent clips
  5. Export to XTTS-v2 format

Usage:
    conda activate new-arabic-tts
    python scripts/refine_cluster.py --top-n 5000
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "egyptian"
WAVS_DIR = DATA_DIR / "wavs"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
BENCHMARKS_DIR = PROJECT_ROOT / "docs" / "benchmarks"

TARGET_SR = 22050
EVAL_SPLIT = 0.1
RANDOM_SEED = 42
SPEAKER_NAME = "egyptian_male_01"
MIN_DURATION = 2.0
MAX_DURATION = 11.0
MIN_TEXT_LEN = 10
MAX_TEXT_LEN = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=5000, help="Number of clips to keep")
    args = parser.parse_args()

    print("=" * 70)
    print("  Speaker Cluster Refinement")
    print(f"  Selecting top {args.top_n} most consistent clips from largest cluster")
    print("=" * 70)
    t_start = time.time()

    # --- Step 1: Load dataset ---
    print("\n[1/6] Loading dataset...")
    ds = load_dataset("MAdel121/arabic-egy-cleaned", split="train")
    print(f"      {len(ds)} clips loaded")

    # --- Step 2: Extract embeddings (reuse ECAPA-TDNN) ---
    print("\n[2/6] Extracting speaker embeddings...")
    from speechbrain.inference.speaker import EncoderClassifier
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda"},
    )

    embeddings = []
    valid_indices = []
    durations = []
    texts = []
    skipped = 0
    t0 = time.time()

    for i in range(len(ds)):
        sample = ds[i]
        audio = sample["audio"]
        duration = sample.get("duration", len(audio["array"]) / audio["sampling_rate"])
        text = sample.get("text", "")

        if duration < MIN_DURATION or duration > MAX_DURATION:
            skipped += 1
            continue
        if len(text) < MIN_TEXT_LEN or len(text) > MAX_TEXT_LEN:
            skipped += 1
            continue

        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        with torch.no_grad():
            emb = classifier.encode_batch(waveform.cuda())
            embeddings.append(emb.squeeze().cpu().numpy())

        valid_indices.append(i)
        durations.append(duration)
        texts.append(text)

        if (i + 1) % 5000 == 0:
            rate = len(valid_indices) / (time.time() - t0)
            eta = (len(ds) - i) / max(rate, 1) / 60
            print(f"      [{i+1:,}/{len(ds):,}] {len(valid_indices):,} valid, "
                  f"{skipped:,} skipped, {rate:.0f} clips/s, ETA {eta:.0f}min")

    embeddings = np.array(embeddings)
    elapsed = time.time() - t0
    print(f"      Done: {len(embeddings):,} valid, {skipped:,} skipped, {elapsed/60:.1f}min")

    # --- Step 3: Find largest cluster ---
    print("\n[3/6] Clustering and selecting largest group...")
    from sklearn.cluster import AgglomerativeClustering

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms

    clustering = AgglomerativeClustering(n_clusters=15)
    labels = clustering.fit_predict(embeddings_norm)

    # Find largest cluster
    unique, counts = np.unique(labels, return_counts=True)
    largest_id = unique[np.argmax(counts)]
    largest_size = counts.max()
    print(f"      Largest cluster: {largest_id} ({largest_size:,} clips)")

    # --- Step 4: Rank by similarity, keep top N ---
    print(f"\n[4/6] Ranking by cosine similarity, keeping top {args.top_n}...")
    mask = labels == largest_id
    cluster_indices = np.where(mask)[0]
    cluster_embs = embeddings_norm[cluster_indices]

    centroid = cluster_embs.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    similarities = cluster_embs @ centroid

    # Sort by similarity descending
    sorted_order = np.argsort(similarities)[::-1]
    top_n = min(args.top_n, len(sorted_order))
    selected_local = sorted_order[:top_n]
    selected_global = cluster_indices[selected_local]
    selected_sims = similarities[selected_local]

    selected_durations = np.array(durations)[selected_global]
    total_hours = selected_durations.sum() / 3600

    print(f"      Selected {top_n:,} clips")
    print(f"      Total hours: {total_hours:.2f}")
    print(f"      Similarity range: {selected_sims[-1]:.4f} — {selected_sims[0]:.4f}")
    print(f"      Mean similarity: {selected_sims.mean():.4f}")
    print(f"      Mean duration: {selected_durations.mean():.2f}s")

    # --- Step 5: Generate visuals ---
    print("\n[5/6] Generating visualizations...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # UMAP
    try:
        from umap import UMAP
        print("      Computing UMAP...")
        max_points = 10000
        # Include all selected + sample of others
        other_indices = np.where(~mask)[0]
        if len(other_indices) > max_points - top_n:
            other_sample = np.random.RandomState(RANDOM_SEED).choice(
                other_indices, max_points - top_n, replace=False)
        else:
            other_sample = other_indices

        viz_indices = np.concatenate([selected_global, other_sample])
        viz_embs = embeddings_norm[viz_indices]
        viz_labels = np.array([1] * len(selected_global) + [0] * len(other_sample))

        reducer = UMAP(n_components=2, random_state=RANDOM_SEED, n_neighbors=30)
        projected = reducer.fit_transform(viz_embs)

        fig, ax = plt.subplots(figsize=(12, 8))
        other_mask = viz_labels == 0
        sel_mask = viz_labels == 1

        ax.scatter(projected[other_mask, 0], projected[other_mask, 1],
                   c="lightgray", s=3, alpha=0.3, label="Other speakers")
        ax.scatter(projected[sel_mask, 0], projected[sel_mask, 1],
                   c="#2ecc71", s=8, alpha=0.6,
                   label=f"Selected: {SPEAKER_NAME} ({top_n:,} clips)")
        ax.set_title("Speaker Embedding Clusters — Refined Selection (UMAP)", fontsize=14)
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

    # Similarity distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(similarities, bins=80, color="#95a5a6", alpha=0.7, label="Full cluster")
    ax.hist(selected_sims, bins=80, color="#2ecc71", alpha=0.7, label=f"Top {top_n:,} selected")
    ax.axvline(x=selected_sims[-1], color="red", linestyle="--", alpha=0.7,
               label=f"Cutoff ({selected_sims[-1]:.3f})")
    ax.set_xlabel("Cosine Similarity to Centroid")
    ax.set_ylabel("Count")
    ax.set_title("Speaker Consistency — Selected vs Full Cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "similarity_distribution.png", dpi=150)
    plt.close(fig)
    print("      Saved similarity_distribution.png")

    # Duration distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(durations, bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0].set_title(f"All Valid Clips ({len(durations):,})")
    axes[0].set_xlabel("Duration (seconds)")
    axes[0].set_ylabel("Count")

    axes[1].hist(selected_durations, bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[1].set_title(f"Selected Speaker ({top_n:,} clips, {total_hours:.1f}h)")
    axes[1].set_xlabel("Duration (seconds)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Audio Duration Distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "duration_distribution.png", dpi=150)
    plt.close(fig)
    print("      Saved duration_distribution.png")

    # Cluster sizes
    fig, ax = plt.subplots(figsize=(10, 5))
    cluster_info = [(u, c) for u, c in zip(unique, counts)]
    cluster_info.sort(key=lambda x: x[1], reverse=True)
    sizes = [c for _, c in cluster_info]
    cids = [f"C{u}" for u, _ in cluster_info]
    colors = ["#2ecc71" if u == largest_id else "#95a5a6" for u, _ in cluster_info]

    ax.bar(range(len(sizes)), sizes, color=colors)
    ax.set_xticks(range(len(cids)))
    ax.set_xticklabels(cids, rotation=45, fontsize=8)
    ax.set_ylabel("Number of Clips")
    ax.set_title(f"Speaker Cluster Sizes (green = source for top {top_n:,} selection)")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "cluster_sizes.png", dpi=150)
    plt.close(fig)
    print("      Saved cluster_sizes.png")

    # --- Step 6: Export ---
    print(f"\n[6/6] Exporting {top_n:,} clips to XTTS-v2 format...")

    # Clean previous export
    if WAVS_DIR.exists():
        import shutil
        shutil.rmtree(WAVS_DIR)
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for j, global_idx in enumerate(selected_global):
        ds_idx = valid_indices[global_idx]
        sample = ds[ds_idx]
        audio = sample["audio"]
        text = sample["text"].strip()

        waveform = torch.tensor(audio["array"], dtype=torch.float32).unsqueeze(0)
        sr = audio["sampling_rate"]
        if sr != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

        wav_filename = f"clip_{j:06d}.wav"
        sf.write(str(WAVS_DIR / wav_filename), waveform.squeeze().numpy(), TARGET_SR)
        rows.append(f"wavs/{wav_filename}|{text}|{SPEAKER_NAME}")

        if (j + 1) % 1000 == 0:
            print(f"      [{j+1:,}/{top_n:,}] exported...")

    # Split
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(rows)
    split_idx = int(len(rows) * (1 - EVAL_SPLIT))

    header = "audio_file|text|speaker_name"
    with open(DATA_DIR / "metadata_train.csv", "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows[:split_idx]) + "\n")
    with open(DATA_DIR / "metadata_eval.csv", "w", encoding="utf-8") as f:
        f.write(header + "\n" + "\n".join(rows[split_idx:]) + "\n")

    elapsed = time.time() - t0
    n_train = split_idx
    n_eval = len(rows) - split_idx

    # Save report
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "source_dataset": "MAdel121/arabic-egy-cleaned",
        "total_source_clips": len(ds),
        "valid_after_filtering": len(valid_indices),
        "clustering": {
            "method": "ECAPA-TDNN + Agglomerative (n=15) + centroid similarity ranking",
            "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
            "largest_cluster_id": int(largest_id),
            "largest_cluster_size": int(largest_size),
            "top_n_selected": top_n,
            "similarity_min": round(float(selected_sims[-1]), 4),
            "similarity_max": round(float(selected_sims[0]), 4),
            "similarity_mean": round(float(selected_sims.mean()), 4),
        },
        "selected": {
            "clips": top_n,
            "total_hours": round(total_hours, 2),
            "mean_duration_s": round(float(selected_durations.mean()), 2),
        },
        "speaker_name": SPEAKER_NAME,
        "export": {
            "train_clips": n_train,
            "eval_clips": n_eval,
            "target_sample_rate": TARGET_SR,
            "format": "XTTS-v2 (pipe-delimited CSV + WAV)",
        },
        "total_pipeline_time_min": round((time.time() - t_start) / 60, 1),
    }
    with open(BENCHMARKS_DIR / "dataset_preparation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  Pipeline Complete!")
    print(f"  Total time: {(time.time() - t_start)/60:.1f} minutes")
    print()
    print(f"  Strategy:  Top {top_n:,} from largest cluster ({largest_size:,} clips)")
    print(f"  Speaker:   {SPEAKER_NAME}")
    print(f"  Train:     {n_train:,} clips")
    print(f"  Eval:      {n_eval:,} clips")
    print(f"  Hours:     {total_hours:.2f}")
    print(f"  Similarity: {selected_sims[-1]:.4f} — {selected_sims[0]:.4f} (mean {selected_sims.mean():.4f})")
    print()
    print(f"  Data:      {DATA_DIR}")
    print(f"  Report:    {BENCHMARKS_DIR / 'dataset_preparation.json'}")
    print(f"  Charts:    {IMAGES_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
