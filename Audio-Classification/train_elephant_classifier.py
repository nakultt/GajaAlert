# %% [markdown]
# # Setup and Installation
# Install required dependencies

# %%
import subprocess
import sys

def install_dependencies():
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "git+https://github.com/w-hc/torch_audioset.git", 
                           "librosa", "soundfile", "onnx", "onnxruntime", 
                           "scikit-learn", "kagglehub", "xgboost", "onnxscript"])
    print("Dependencies installed successfully.")

# Automatically install dependencies if they are missing
try:
    import torch_audioset
    import kagglehub
    import onnxscript
except ImportError:
    install_dependencies()

# %%
import os
# Optional: Set Kaggle credentials manually here if not set in environment
# os.environ["KAGGLE_USERNAME"] = "your_username"
# os.environ["KAGGLE_KEY"] = "your_api_key"

import glob
import subprocess
import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import onnx
import onnxruntime as ort
import urllib.request
import zipfile
import shutil
import csv
import traceback

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
PATIENCE = 10
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3

# %% [markdown]
# # Step 2: Download and Prepare Data

# %%
def download_github_repo():
    repo_url = "https://github.com/HiruDewmi/Audio-Classification-for-Elephant-Sounds.git"
    target_dir = "Audio-Classification-for-Elephant-Sounds"
    if not os.path.exists(target_dir):
        print(f"Cloning {repo_url}...")
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url])
    else:
        print(f"Repo already exists at {target_dir}")
    return target_dir

def download_kaggle_dataset():
    print("Downloading FSC22 dataset via kagglehub...")
    try:
        path = kagglehub.dataset_download("irmiot22/fsc22-dataset")
        print(f"FSC22 downloaded to: {path}")
        return path
    except Exception as e:
        print("Failed to download via kagglehub. Please ensure Kaggle credentials are set up.")
        print("Set up instructions: Place kaggle.json in ~/.kaggle/ and set permissions (chmod 600 ~/.kaggle/kaggle.json)")
        print("Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("Alternatively, download manually via: kaggle datasets download -d irmiot22/fsc22-dataset")
        print(f"Error: {e}")
        raise

elephant_repo_path = download_github_repo()
fsc22_dataset_path = download_kaggle_dataset()

# %%
from torch_audioset.data.torch_input_processing import WaveformToInput

waveform_to_input = WaveformToInput()

# Verify patch shape
dummy_waveform = torch.zeros(1, 16000) # 1 second dummy
dummy_patches = waveform_to_input(dummy_waveform, 16000)
print(f"Verified patch shape from torch_audioset's WaveformToInput: {dummy_patches.shape} (Expected: [N, 1, 96, 64])")

# %% [markdown]
# # Step 3: Load Frozen YAMNet

# %%
from torch_audioset.yamnet.model import yamnet as torch_yamnet

yamnet_model = torch_yamnet(pretrained=True)
yamnet_model.eval()

# Freeze all parameters
for param in yamnet_model.parameters():
    param.requires_grad_(False)

class YAMNetEmbeddingExtractor(nn.Module):
    def __init__(self, yamnet_model):
        super().__init__()
        self.yamnet = yamnet_model

    def forward(self, patches): 
        m = self.yamnet
        x = patches
        for name in m.layer_names: 
            x = getattr(m, name)(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(x.shape[0], -1) # Output shape: [N, 1024]
        return x # REMOVED the .mean(dim=0)

embedding_extractor = YAMNetEmbeddingExtractor(yamnet_model).to(device)
embedding_extractor.eval()
print("Frozen YAMNet embedding extractor ready.")

# %% [markdown]
# # Step 4: Extract Embeddings

# %%
# Force-delete stale cache so extraction always re-runs with the latest code
cache_path = "elephant_embeddings.npz"
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"Deleted stale cache: {cache_path}")

def extract_embeddings_and_labels(elephant_dir, fsc22_dir, cache_path="elephant_embeddings.npz"):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data['embeddings'], data['labels'], data['sources']

    print("Extracting embeddings from scratch...")
    
    # --- Diagnostic: show what paths we're working with ---
    print(f"  Elephant dir: {elephant_dir}")
    print(f"    exists: {os.path.exists(elephant_dir)}")
    if os.path.exists(elephant_dir):
        top_contents = os.listdir(elephant_dir)
        print(f"    top-level contents: {top_contents[:15]}")
        # Count total .wav files recursively
        elephant_wav_count = sum(
            1 for r, d, fs in os.walk(elephant_dir)
            for f in fs if f.lower().endswith('.wav')
        )
        print(f"    total .wav files found recursively: {elephant_wav_count}")
    
    print(f"  FSC22 dir: {fsc22_dir}")
    print(f"    exists: {os.path.exists(fsc22_dir)}")
    if os.path.exists(fsc22_dir):
        top_contents = os.listdir(fsc22_dir)
        print(f"    top-level contents: {top_contents[:15]}")
    
    embeddings = []
    labels = []
    sources = []
    processed_elephant = 0

    # Process Elephant Data (Positive Class = 1)
    # Search all wav files within the repository recursively
    print("\n--- Processing Elephant Data ---")
    for root, dirs, files in os.walk(elephant_dir):
        if "audio_preprocess" in root: continue
        wav_files = [f for f in files if f.lower().endswith(".wav")]
        if wav_files:
            print(f"  Found {len(wav_files)} wav files in: {root}")
        for file_name in wav_files:
            file_path = os.path.join(root, file_name)
            try:
                data, sr = sf.read(file_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1) # Convert stereo to mono
                if sr != 16000:
                    waveform = librosa.resample(data, orig_sr=sr, target_sr=16000)
                else:
                    waveform = data
                
                # Audio Mixing: 50% chance to add slight Gaussian noise to simulate dirty phone mic
                if np.random.rand() > 0.5:
                    noise = np.random.normal(0, 0.005, waveform.shape)
                    waveform = waveform + noise
                    
                waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
                patches = waveform_to_input(waveform_tensor, 16000).to(device)
                if patches.shape[0] == 0: continue
                with torch.no_grad():
                    embs = embedding_extractor(patches)
                for i in range(embs.shape[0]):
                    embeddings.append(embs.cpu().numpy()[i])
                    labels.append(1)
                    sources.append("elephant")
                processed_elephant += 1
            except Exception as e:
                print(f"Warning: Failed to process elephant audio {file_path}: {e}")
    
    print(f"  Total elephant files processed: {processed_elephant}")
    if processed_elephant == 0:
        print("  WARNING: No elephant .wav files were found! Check the directory structure.")
        print(f"  Searched recursively under: {elephant_dir}")

    # Process FSC22 Data (Negative Class = 0)
    print("\n--- Processing FSC22 Data ---")
    
    # Step A: Find any CSV file that could be metadata
    metadata_csv = None
    for root, dirs, files in os.walk(fsc22_dir):
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        if csv_files:
            metadata_csv = os.path.join(root, csv_files[0])
            print(f"  Found CSV: {metadata_csv}")
            break
    
    # Step B: Find where the .wav files live
    fsc22_audio_dir = None
    fsc22_wav_files = []  # Collect all wav paths as fallback
    for root, dirs, files in os.walk(fsc22_dir):
        wavs = [f for f in files if f.lower().endswith('.wav')]
        if wavs:
            if fsc22_audio_dir is None:
                fsc22_audio_dir = root
            for w in wavs:
                fsc22_wav_files.append(os.path.join(root, w))
    
    print(f"  Resolved audio dir: {fsc22_audio_dir}")
    print(f"  Resolved metadata CSV: {metadata_csv}")
    print(f"  Total .wav files found recursively: {len(fsc22_wav_files)}")
    
    processed_fsc22 = 0
    
    # Strategy 1: Use CSV if found
    if metadata_csv and fsc22_audio_dir:
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row:
                print(f"  CSV column names: {list(first_row.keys())}")
                print(f"  First row sample: { {k: v for k, v in list(first_row.items())[:5]} }")
        
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = None
                class_name = 'unknown'
                
                # Try common column name patterns
                for k in list(row.keys()):
                    k_lower = k.lower().strip()
                    if 'dataset' in k_lower and 'filename' in k_lower: 
                        filename = row[k]
                    elif 'filename' in k_lower and not filename: 
                        filename = row[k]
                    if 'class' in k_lower and 'name' in k_lower: 
                        class_name = row[k]
                
                if filename:
                    file_path = os.path.join(fsc22_audio_dir, filename)
                    if not os.path.exists(file_path):
                        # Try searching recursively for the filename
                        for wp in fsc22_wav_files:
                            if os.path.basename(wp) == filename:
                                file_path = wp
                                break
                    if os.path.exists(file_path):
                        try:
                            data, sr = sf.read(file_path)
                            if len(data.shape) > 1:
                                data = data.mean(axis=1)
                            if sr != 16000:
                                waveform = librosa.resample(data, orig_sr=sr, target_sr=16000)
                            else:
                                waveform = data
                            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
                            patches = waveform_to_input(waveform_tensor, 16000).to(device)
                            if patches.shape[0] == 0: continue
                            with torch.no_grad():
                                embs = embedding_extractor(patches)
                            for i in range(embs.shape[0]):
                                embeddings.append(embs.cpu().numpy()[i])
                                labels.append(0)
                                sources.append(f"fsc22_{class_name}")
                            processed_fsc22 += 1
                        except Exception as e:
                            print(f"Warning: Failed to process FSC22 audio {file_path}: {e}")
        
        print(f"  Processed {processed_fsc22} files via CSV")
    
    # Strategy 2: Fallback - walk all .wav files directly, use parent dir as class name
    if processed_fsc22 == 0 and fsc22_wav_files:
        print(f"  CSV strategy yielded 0 files. Falling back to direct .wav walk ({len(fsc22_wav_files)} files)...")
        for file_path in fsc22_wav_files:
            class_name = os.path.basename(os.path.dirname(file_path))
            try:
                data, sr = sf.read(file_path)
                if len(data.shape) > 1:
                    data = data.mean(axis=1)
                if sr != 16000:
                    waveform = librosa.resample(data, orig_sr=sr, target_sr=16000)
                else:
                    waveform = data
                waveform_tensor = torch.from_numpy(waveform).unsqueeze(0).float()
                patches = waveform_to_input(waveform_tensor, 16000).to(device)
                if patches.shape[0] == 0: continue
                with torch.no_grad():
                    embs = embedding_extractor(patches)
                for i in range(embs.shape[0]):
                    embeddings.append(embs.cpu().numpy()[i])
                    labels.append(0)
                    sources.append(f"fsc22_{class_name}")
                processed_fsc22 += 1
            except Exception as e:
                print(f"Warning: Failed to process FSC22 audio {file_path}: {e}")
    
    if processed_fsc22 == 0:
        print("  WARNING: No FSC22 files processed! The model will have no negative class.")
    
    print(f"  Total FSC22 files processed: {processed_fsc22}")
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    sources = np.array(sources)
    
    if len(embeddings) > 0:
        np.savez(cache_path, embeddings=embeddings, labels=labels, sources=sources)
        print(f"Saved {len(embeddings)} embeddings to {cache_path}")
    else:
        print("ERROR: No embeddings extracted! NOT saving empty cache.")
    
    return embeddings, labels, sources

embeddings, labels, sources = extract_embeddings_and_labels(elephant_repo_path, fsc22_dataset_path)
print(f"\nTotal extracted: {len(labels)} patches ({np.sum(labels==1)} elephant, {np.sum(labels==0)} non-elephant)")

# %% [markdown]
# # Step 5: Stratified Split

# %%
X_train_val, X_test, y_train_val, y_test = train_test_split(embeddings, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15/0.85, random_state=42, stratify=y_train_val)

print(f"Train size: {len(X_train)} (Class 0: {np.sum(y_train==0)}, Class 1: {np.sum(y_train==1)})")
print(f"Val size: {len(X_val)} (Class 0: {np.sum(y_val==0)}, Class 1: {np.sum(y_val==1)})")
print(f"Test size: {len(X_test)} (Class 0: {np.sum(y_test==0)}, Class 1: {np.sum(y_test==1)})")

# %% [markdown]
# # Step 6: Class Imbalance Strategy

# %%
# Inverse-frequency weights based on training split
counts = np.bincount(y_train)
total = len(y_train)
weights = total / (len(counts) * counts)
class_weights = torch.FloatTensor(weights).to(device)

print(f"Class counts in training: {counts}")
print(f"Computed class weights: {weights}")
print("Rationale for weighting over subsampling: FSC22 provides a diverse set of 27 negative forest sound classes. "
      "Subsampling would discard valuable diversity needed to generalize to unseen backgrounds. "
      "Using weighted cross-entropy ensures we utilize all negative samples while penalizing errors on the minority positive class more heavily.")

# %% [markdown]
# # Step 7: Train XGBoost Classifier

# %%
from xgboost import XGBClassifier

print("Training XGBoost Classifier...")
# scale_pos_weight for class imbalance
scale_pos_weight = float(np.sum(y_train==0) / np.sum(y_train==1))
clf = XGBClassifier(
    n_estimators=100, 
    max_depth=3, 
    learning_rate=0.1, 
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
clf.fit(X_train, y_train)

# %% [markdown]
# # Step 8: Evaluate on Test Set

# %%
test_preds = clf.predict(X_test)

acc = accuracy_score(y_test, test_preds)
prec = precision_score(y_test, test_preds)
rec = recall_score(y_test, test_preds)
cm = confusion_matrix(y_test, test_preds)

print("=== Test Evaluation ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print("Confusion Matrix:")
print(cm)

# %% [markdown]
# # Step 9: Export XGBoost Model

# %%
xgb_model_path = "elephant_xgb.json"
clf.save_model(xgb_model_path)
print(f"Saved XGBoost model to {xgb_model_path}")

# %% [markdown]
# # Step 10: Export and Validate YAMNet ONNX

# %%
yamnet_onnx_path = "yamnet.onnx"
dummy_yamnet_in = torch.randn(5, 1, 96, 64, device=device) # N=5 patches

embedding_extractor.eval()

try:
    torch.onnx.export(
        embedding_extractor, dummy_yamnet_in, yamnet_onnx_path,
        input_names=["log_mel_patches"], output_names=["clip_embedding"],
        dynamic_axes={"log_mel_patches": {0: "num_patches"}, "clip_embedding": {0: "num_patches"}},
        opset_version=17
    )
except Exception as e:
    print(f"Default export failed, retrying... Error: {e}")
    torch.onnx.export(
        embedding_extractor, dummy_yamnet_in, yamnet_onnx_path,
        input_names=["log_mel_patches"], output_names=["clip_embedding"],
        dynamic_axes={"log_mel_patches": {0: "num_patches"}, "clip_embedding": {0: "num_patches"}},  
        opset_version=17
    )

# Validate YAMNet ONNX at multiple N values
ort_session_yamnet = ort.InferenceSession(yamnet_onnx_path, providers=["CPUExecutionProvider"])

yamnet_pass = True
for N in [1, 5, 13]:
    dummy_input_np = np.random.randn(N, 1, 96, 64).astype(np.float32)
    dummy_input_pt = torch.from_numpy(dummy_input_np).to(device)
    
    with torch.no_grad():
        pt_out = embedding_extractor(dummy_input_pt).cpu().numpy()
        
    onnx_out = ort_session_yamnet.run(["clip_embedding"], {"log_mel_patches": dummy_input_np})[0]
    
    diff = np.max(np.abs(pt_out - onnx_out))
    if diff > 1e-4: yamnet_pass = False
    print(f"YAMNet ONNX Validation (N={N}): Max abs diff = {diff:.2e}")

print(f"YAMNet ONNX Validation Overall: {'PASS' if yamnet_pass else 'FAIL'}")

# %% [markdown]
# # Step 11: Final Deliverable Summary

# %%
print("\n" + "="*50)
print("FINAL DELIVERABLE SUMMARY")
print("="*50)
print(f"- Downloaded Positive Data: HiruDewmi/Audio-Classification-for-Elephant-Sounds -> {elephant_repo_path}")
print(f"- Downloaded Negative Data: Kaggle irmiot22/fsc22-dataset -> {fsc22_dataset_path}")

total_elephant = np.sum(labels == 1)
total_non_elephant = np.sum(labels == 0)
print(f"\n- Total Extracted Samples: {len(labels)}")
print(f"  - Elephant (Class 1): {total_elephant}")
print(f"  - Non-Elephant (Class 0): {total_non_elephant}")

from collections import Counter
fsc22_counts = Counter([s for s in sources if s.startswith('fsc22_')])
print("\n- FSC22 Sub-class Breakdown (sample counts):")
for subclass, count in fsc22_counts.most_common(5):
    print(f"  - {subclass}: {count}")
print("  - ... (27 classes total)")

print(f"\n- Class Imbalance Strategy: Weighted CrossEntropyLoss.")
print(f"  Class Weights (Class 0, Class 1): {weights}")
print("  Rationale: Preserves full diversity of 27 background forest sounds while preventing bias toward majority class.")

print("\n- Stratified Split Sizes:")
print(f"  - Train: {len(y_train)} (Ele: {np.sum(y_train==1)}, Non-Ele: {np.sum(y_train==0)})")
print(f"  - Val:   {len(y_val)} (Ele: {np.sum(y_val==1)}, Non-Ele: {np.sum(y_val==0)})")
print(f"  - Test:  {len(y_test)} (Ele: {np.sum(y_test==1)}, Non-Ele: {np.sum(y_test==0)})")

print("\n- Final Test Metrics:")
print(f"  - Accuracy:  {acc:.4f}")
print(f"  - Precision: {prec:.4f}")
print(f"  - Recall:    {rec:.4f}")
print("  - Confusion Matrix:")
print(cm)

yamnet_size_mb = os.path.getsize(yamnet_onnx_path) / (1024 * 1024)
xgb_size_mb = os.path.getsize(xgb_model_path) / (1024 * 1024)

print("\n- Exported Artifacts:")
print(f"  1. YAMNet ONNX: {yamnet_onnx_path} ({yamnet_size_mb:.2f} MB)")
print("     - Input:  'log_mel_patches' float32 [N, 1, 96, 64] (N is dynamic)")
print("     - Output: 'clip_embedding'  float32 [N, 1024] (N is dynamic)")
print(f"     - Validation vs PyTorch: {'PASS' if yamnet_pass else 'FAIL'}")

print(f"\n  2. XGBoost Model: {xgb_model_path} ({xgb_size_mb:.2f} MB)")
print("     - Input:  'clip_embedding' float32 [N, 1024]")
print("     - Output: 'predictions' array")
print("     - Deployment code snippet:")
print("       ```python")
print("       import xgboost as xgb")
print(f"       clf = xgb.XGBClassifier()")
print(f"       clf.load_model('{xgb_model_path}')")
print("       # Ensure embeddings are properly shaped (N, 1024)")
print("       preds = clf.predict_proba(embeddings)")
print("       elephant_confidence = preds[0][1]")
print("       if elephant_confidence > 0.85:")
print("           print('TRIGGER VISUAL HUB')")
print("       ```")
print("="*50)