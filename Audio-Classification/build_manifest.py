#!/usr/bin/env python3
"""
build_manifest.py — Build a leak-free dataset manifest for elephant binary detection.

Colab-optimized: auto-installs deps, mounts Drive, handles Kaggle auth.
Run in Colab:  !python build_manifest.py
Run locally:   python build_manifest.py --elephant-dir <path> --negative-dir <path>

Outputs:
  dataset/manifest.jsonl         — one JSON object per audio file
  dataset/manifest_summary.json  — counts, class balance, duplicate report
"""

import os
import sys
import json
import hashlib
import re
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

# ── Colab auto-setup ────────────────────────────────────────────────────────────

def is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

def colab_setup():
    """Auto-install deps and mount Drive when running in Colab."""
    if not is_colab():
        return

    print("[Colab] Installing dependencies...")
    os.system("pip install -q librosa soundfile kagglehub")

    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
        print("[Colab] Drive mounted.")

colab_setup()

import numpy as np

try:
    import soundfile as sf
except ImportError:
    os.system("pip install soundfile")
    import soundfile as sf

try:
    import librosa
except ImportError:
    os.system("pip install librosa")
    import librosa


# ── Constants ───────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
ELEPHANT_KEYWORDS = {"roar", "rumble", "trumpet", "elephant"}
SEED = 42

# Regex patterns for augmentation detection
PADDED_RE = re.compile(r"^padded_(.+)$", re.IGNORECASE)
# e.g. Roar100.wav -> augmented from Roar (original range 01-99, augmented 100+)
HIGH_NUM_RE = re.compile(r"^(Roar|Rumble|Trumpet)(\d+)\.wav$", re.IGNORECASE)


# ── Utilities ───────────────────────────────────────────────────────────────────

def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_audio_info(path: str) -> dict:
    """Get duration and sample rate without loading the full waveform."""
    try:
        info = sf.info(path)
        return {"duration_s": info.duration, "sample_rate": info.samplerate, "ok": True}
    except Exception as e:
        # Fallback via librosa
        try:
            duration = librosa.get_duration(path=path)
            y, sr = librosa.load(path, sr=None, duration=0.01)
            return {"duration_s": duration, "sample_rate": sr, "ok": True}
        except Exception:
            return {"duration_s": 0.0, "sample_rate": 0, "ok": False, "error": str(e)}


def detect_augmentation_parent(filename: str) -> str | None:
    """
    Heuristic: detect if a file is an augmentation of another.
    Returns the inferred original base name, or None.
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix

    # padded_Roar05.wav -> Roar05.wav
    m = PADDED_RE.match(stem)
    if m:
        return m.group(1) + ext

    # Roar100.wav -> augmented (originals are typically 01-~50)
    m = HIGH_NUM_RE.match(filename)
    if m:
        prefix, num_str = m.group(1), m.group(2)
        num = int(num_str)
        if num >= 100:
            # This is an augmented file; group with the prefix
            return f"{prefix}_augmented_group"

    return None


def assign_recording_group(entries: list[dict]) -> list[dict]:
    """
    Assign a recording_group ID to each entry.
    Files with the same SHA-256 hash share a group.
    Files detected as augmentations share their parent's group.
    """
    # Step 1: group by hash to find exact duplicates
    hash_to_group = {}
    group_counter = 0

    # Sort for determinism
    entries.sort(key=lambda e: e["path"])

    # First pass: assign groups by hash
    for entry in entries:
        h = entry["sha256"]
        if h not in hash_to_group:
            hash_to_group[h] = group_counter
            group_counter += 1
        entry["recording_group"] = hash_to_group[h]

    # Step 2: merge groups for augmented files
    # Build a map: directory+base_name -> group
    dir_base_to_group = {}
    merge_map = {}  # group_id -> canonical_group_id

    for entry in entries:
        dirpath = os.path.dirname(entry["path"])
        filename = os.path.basename(entry["path"])
        parent = detect_augmentation_parent(filename)

        if parent is not None:
            key = (dirpath, parent)
        else:
            key = (dirpath, filename)

        if key in dir_base_to_group:
            # Merge this entry's group with the existing group
            existing_group = dir_base_to_group[key]
            current_group = entry["recording_group"]
            if existing_group != current_group:
                # Always merge to the lower group number
                lo, hi = min(existing_group, current_group), max(existing_group, current_group)
                merge_map[hi] = lo
        else:
            dir_base_to_group[key] = entry["recording_group"]

    # Resolve transitive merges
    def resolve(g):
        visited = set()
        while g in merge_map and g not in visited:
            visited.add(g)
            g = merge_map[g]
        return g

    for entry in entries:
        entry["recording_group"] = resolve(entry["recording_group"])

    # Re-number groups to be contiguous
    unique_groups = sorted(set(e["recording_group"] for e in entries))
    remap = {old: new for new, old in enumerate(unique_groups)}
    for entry in entries:
        entry["recording_group"] = remap[entry["recording_group"]]

    return entries


def stratified_group_split(
    entries: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = SEED,
) -> list[dict]:
    """
    Split entries at the recording_group level, stratified by label.
    No recording group crosses splits.
    """
    rng = np.random.RandomState(seed)

    # Collect groups with their label
    group_to_label = {}
    group_to_entries = defaultdict(list)
    for entry in entries:
        g = entry["recording_group"]
        group_to_entries[g].append(entry)
        # A group's label is the majority label (should be unanimous)
        group_to_label[g] = entry["label"]

    # Separate groups by label
    pos_groups = [g for g, l in group_to_label.items() if l == "elephant"]
    neg_groups = [g for g, l in group_to_label.items() if l == "non-elephant"]

    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    def split_list(items, fracs):
        n = len(items)
        n_train = max(1, int(n * fracs[0]))
        n_val = max(1, int(n * fracs[1]))
        # Remainder goes to test
        return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]

    pos_train, pos_val, pos_test = split_list(pos_groups, [train_frac, val_frac])
    neg_train, neg_val, neg_test = split_list(neg_groups, [train_frac, val_frac])

    split_map = {}
    for g in pos_train + neg_train:
        split_map[g] = "train"
    for g in pos_val + neg_val:
        split_map[g] = "val"
    for g in pos_test + neg_test:
        split_map[g] = "test"

    for entry in entries:
        entry["split"] = split_map[entry["recording_group"]]

    return entries


# ── Main manifest builder ──────────────────────────────────────────────────────

def scan_elephant_dir(elephant_dir: str) -> list[dict]:
    """Walk elephant audio directory and build entries."""
    entries = []
    elephant_dir = os.path.abspath(elephant_dir)

    if not os.path.isdir(elephant_dir):
        print(f"[WARN] Elephant directory not found: {elephant_dir}")
        return entries

    print(f"[Phase 1] Scanning elephant directory: {elephant_dir}")
    count = 0

    for root, dirs, files in os.walk(elephant_dir):
        # Skip preprocessing/augmentation tool directories
        if "audio_preprocess" in root or "__pycache__" in root:
            continue
        for fname in sorted(files):
            if Path(fname).suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            rel_dir = os.path.basename(os.path.dirname(fpath))  # e.g. Roar, Rumble
            source_class = rel_dir if rel_dir.lower() in ELEPHANT_KEYWORDS else "elephant_unknown"

            info = get_audio_info(fpath)
            if not info["ok"]:
                print(f"  [SKIP] Cannot read: {fpath} — {info.get('error', 'unknown')}")
                continue

            entry = {
                "path": fpath,
                "filename": fname,
                "label": "elephant",
                "source_class": source_class.lower(),
                "duration_s": round(info["duration_s"], 3),
                "sample_rate": info["sample_rate"],
                "sha256": sha256_file(fpath),
                "augmentation_parent": detect_augmentation_parent(fname),
            }
            entries.append(entry)
            count += 1

    print(f"  Found {count} elephant audio files.")
    return entries


def scan_negative_dir(negative_dir: str) -> list[dict]:
    """Walk negative audio directory and build entries."""
    entries = []
    negative_dir = os.path.abspath(negative_dir)

    if not os.path.isdir(negative_dir):
        print(f"[WARN] Negative directory not found: {negative_dir}")
        return entries

    print(f"[Phase 1] Scanning negative directory: {negative_dir}")
    count = 0

    for root, dirs, files in os.walk(negative_dir):
        if "__pycache__" in root:
            continue
        for fname in sorted(files):
            if Path(fname).suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            fpath = os.path.join(root, fname)
            # Use parent directory name as class label
            parent_dir = os.path.basename(os.path.dirname(fpath))

            info = get_audio_info(fpath)
            if not info["ok"]:
                continue

            entry = {
                "path": fpath,
                "filename": fname,
                "label": "non-elephant",
                "source_class": parent_dir.lower(),
                "duration_s": round(info["duration_s"], 3),
                "sample_rate": info["sample_rate"],
                "sha256": sha256_file(fpath),
                "augmentation_parent": detect_augmentation_parent(fname),
            }
            entries.append(entry)
            count += 1

            if count % 200 == 0:
                print(f"  Scanned {count} negative files...")

    print(f"  Found {count} negative audio files.")
    return entries


def find_duplicates(entries: list[dict]) -> dict:
    """Find byte-identical files across the dataset."""
    hash_groups = defaultdict(list)
    for e in entries:
        hash_groups[e["sha256"]].append(e["path"])

    duplicates = {h: paths for h, paths in hash_groups.items() if len(paths) > 1}
    return duplicates


def deduplicate(entries: list[dict]) -> list[dict]:
    """
    Remove byte-identical duplicates, keeping one representative per hash.
    Prefer files in train/ over validate/ over test/ to maximise training data.
    """
    hash_seen = {}
    deduped = []
    removed = 0

    # Sort: prefer paths containing 'train', then 'validate', then others
    def sort_key(e):
        p = e["path"].lower()
        if "train" in p:
            return (0, p)
        elif "validate" in p or "val" in p:
            return (1, p)
        else:
            return (2, p)

    entries_sorted = sorted(entries, key=sort_key)

    for entry in entries_sorted:
        h = entry["sha256"]
        if h not in hash_seen:
            hash_seen[h] = entry["path"]
            deduped.append(entry)
        else:
            removed += 1

    if removed > 0:
        print(f"  Removed {removed} byte-identical duplicates.")
    return deduped


def build_manifest(elephant_dir: str, negative_dir: str, output_dir: str = "dataset"):
    """Build the complete manifest."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("BUILD MANIFEST — Leak-Free Dataset for Elephant Binary Detection")
    print("=" * 70)

    # Scan directories
    elephant_entries = scan_elephant_dir(elephant_dir)
    negative_entries = scan_negative_dir(negative_dir)

    all_entries = elephant_entries + negative_entries

    if len(all_entries) == 0:
        print("[ERROR] No audio files found. Check your --elephant-dir and --negative-dir paths.")
        sys.exit(1)

    # Find and report duplicates
    duplicates = find_duplicates(all_entries)
    if duplicates:
        print(f"\n[Phase 1] Found {len(duplicates)} sets of byte-identical files:")
        for h, paths in list(duplicates.items())[:10]:
            print(f"  SHA-256 {h[:16]}...:")
            for p in paths:
                print(f"    - {p}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicate sets.")

    # Deduplicate
    all_entries = deduplicate(all_entries)

    # Assign recording groups
    print(f"\n[Phase 1] Assigning recording groups...")
    all_entries = assign_recording_group(all_entries)
    n_groups = len(set(e["recording_group"] for e in all_entries))
    print(f"  {len(all_entries)} files -> {n_groups} recording groups")

    # Split
    print(f"\n[Phase 1] Splitting by recording group (70/15/15)...")
    all_entries = stratified_group_split(all_entries)

    # ── Acceptance gate ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ACCEPTANCE GATE")
    print(f"{'='*70}")

    # Check 1: every file in exactly one split
    split_counts = Counter(e["split"] for e in all_entries)
    print(f"  Split distribution: {dict(split_counts)}")
    assert sum(split_counts.values()) == len(all_entries), "File count mismatch!"

    # Check 2: no recording group crosses splits
    group_splits = defaultdict(set)
    for e in all_entries:
        group_splits[e["recording_group"]].add(e["split"])

    leaked_groups = {g: splits for g, splits in group_splits.items() if len(splits) > 1}
    if leaked_groups:
        print(f"  [FAIL] {len(leaked_groups)} recording groups cross splits!")
        for g, splits in list(leaked_groups.items())[:5]:
            print(f"    Group {g}: {splits}")
        sys.exit(1)
    else:
        print("  [PASS] No recording group crosses splits.")

    # Check 3: no SHA-256 crosses splits
    hash_splits = defaultdict(set)
    for e in all_entries:
        hash_splits[e["sha256"]].add(e["split"])
    leaked_hashes = {h: s for h, s in hash_splits.items() if len(s) > 1}
    if leaked_hashes:
        print(f"  [FAIL] {len(leaked_hashes)} hashes appear in multiple splits!")
        sys.exit(1)
    else:
        print("  [PASS] No hash crosses splits.")

    # Check 4: both classes present in all splits
    for split_name in ["train", "val", "test"]:
        split_entries = [e for e in all_entries if e["split"] == split_name]
        labels_in_split = set(e["label"] for e in split_entries)
        if len(labels_in_split) < 2:
            print(f"  [WARN] Split '{split_name}' has only labels: {labels_in_split}")
        else:
            ele_count = sum(1 for e in split_entries if e["label"] == "elephant")
            neg_count = sum(1 for e in split_entries if e["label"] == "non-elephant")
            print(f"  [PASS] Split '{split_name}': {ele_count} elephant, {neg_count} non-elephant")

    print(f"\n  ACCEPTANCE GATE: PASSED ✓")

    # ── Write manifest ──────────────────────────────────────────────────────
    manifest_path = os.path.join(output_dir, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n[Phase 1] Wrote {len(all_entries)} entries to {manifest_path}")

    # ── Write summary ───────────────────────────────────────────────────────
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_files": len(all_entries),
        "total_recording_groups": n_groups,
        "duplicates_removed": len(elephant_entries) + len(negative_entries) - len(all_entries),
        "duplicate_sets": len(duplicates),
        "splits": {},
        "label_distribution": dict(Counter(e["label"] for e in all_entries)),
        "source_class_distribution": dict(Counter(e["source_class"] for e in all_entries)),
        "duration_stats": {
            "total_seconds": round(sum(e["duration_s"] for e in all_entries), 1),
            "mean_seconds": round(np.mean([e["duration_s"] for e in all_entries]), 2),
            "min_seconds": round(min(e["duration_s"] for e in all_entries), 2),
            "max_seconds": round(max(e["duration_s"] for e in all_entries), 2),
        },
    }
    for split_name in ["train", "val", "test"]:
        split_entries = [e for e in all_entries if e["split"] == split_name]
        summary["splits"][split_name] = {
            "total": len(split_entries),
            "elephant": sum(1 for e in split_entries if e["label"] == "elephant"),
            "non_elephant": sum(1 for e in split_entries if e["label"] == "non-elephant"),
            "recording_groups": len(set(e["recording_group"] for e in split_entries)),
        }

    summary_path = os.path.join(output_dir, "manifest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Phase 1] Wrote summary to {summary_path}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(json.dumps(summary, indent=2))

    return manifest_path, summary


# ── Kaggle / dataset helpers ────────────────────────────────────────────────────

def ensure_elephant_data(base_dir: str) -> str:
    """Clone elephant dataset if not present."""
    # Check common locations
    candidates = [
        os.path.join(base_dir, "Audio-Classification-for-Elephant-Sounds", "data"),
        os.path.join(base_dir, "..", "ElephantCallerNet",
                     "Audio-Classification-for-Elephant-Sounds", "data"),
    ]
    if is_colab():
        candidates.insert(0, "/content/Audio-Classification-for-Elephant-Sounds/data")
        candidates.insert(0, "/content/drive/MyDrive/Qualcomm Hackathon/"
                          "ElephantCallerNet/Audio-Classification-for-Elephant-Sounds/data")

    for path in candidates:
        if os.path.isdir(path):
            wav_count = sum(1 for _, _, fs in os.walk(path)
                          for f in fs if f.lower().endswith(".wav"))
            if wav_count > 0:
                print(f"[Data] Found elephant data at: {path} ({wav_count} WAV files)")
                return path

    # Clone if not found
    print("[Data] Elephant dataset not found locally. Cloning...")
    clone_dir = os.path.join(base_dir, "Audio-Classification-for-Elephant-Sounds")
    if not os.path.isdir(clone_dir):
        os.system("git clone --depth 1 "
                  "https://github.com/HiruDewmi/Audio-Classification-for-Elephant-Sounds.git "
                  f'"{clone_dir}"')
    data_dir = os.path.join(clone_dir, "data")
    if os.path.isdir(data_dir):
        return data_dir
    # Fallback: maybe WAVs are in the root
    return clone_dir


def ensure_negative_data(base_dir: str) -> str:
    """Download FSC22 via kagglehub if credentials are available, else use ESC-50."""
    # Check if FSC22 already exists
    candidates = []
    if is_colab():
        candidates.append("/content/fsc22")
        candidates.append("/content/drive/MyDrive/datasets/fsc22")

    for path in candidates:
        if os.path.isdir(path):
            print(f"[Data] Found negative data at: {path}")
            return path

    # Try kagglehub
    try:
        import kagglehub
        print("[Data] Downloading FSC22 via kagglehub...")
        path = kagglehub.dataset_download("irmiot22/fsc22-dataset")
        print(f"[Data] FSC22 downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[Data] Kaggle download failed: {e}")

    # Fallback: try ESC-50
    try:
        import kagglehub
        print("[Data] Trying ESC-50 as fallback...")
        path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
        print(f"[Data] ESC-50 downloaded to: {path}")
        return path
    except Exception as e:
        print(f"[Data] ESC-50 download also failed: {e}")
        print("[Data] Please provide --negative-dir manually.")
        sys.exit(1)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build a leak-free dataset manifest for elephant binary detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect data (Colab or local):
  python build_manifest.py

  # Explicit paths:
  python build_manifest.py \\
    --elephant-dir ../ElephantCallerNet/Audio-Classification-for-Elephant-Sounds/data \\
    --negative-dir /path/to/fsc22

  # Custom output:
  python build_manifest.py --output-dir my_dataset
        """,
    )
    parser.add_argument("--elephant-dir", type=str, default=None,
                        help="Path to elephant WAV files (Roar/Rumble/Trumpet subdirs)")
    parser.add_argument("--negative-dir", type=str, default=None,
                        help="Path to negative audio files (FSC22, ESC-50, etc.)")
    parser.add_argument("--output-dir", type=str, default="dataset",
                        help="Output directory for manifest files (default: dataset/)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Resolve data directories
    elephant_dir = args.elephant_dir or ensure_elephant_data(script_dir)
    negative_dir = args.negative_dir or ensure_negative_data(script_dir)

    build_manifest(elephant_dir, negative_dir, args.output_dir)


if __name__ == "__main__":
    main()
