import kagglehub
import shutil
import os

# Folder where you want the datasets
SAVE_DIR = r"C:\Users\nisha\OneDrive\Desktop\yamnet\dataset"

# Create the folder if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- ESC-50 ----------------
print("Downloading ESC-50...")
esc50_cache = kagglehub.dataset_download(
    "mmoreaux/environmental-sound-classification-50"
)

esc50_dest = os.path.join(SAVE_DIR, "ESC50")

if os.path.exists(esc50_dest):
    shutil.rmtree(esc50_dest)

shutil.copytree(esc50_cache, esc50_dest)

print(f"ESC-50 saved to: {esc50_dest}")

# ---------------- UrbanSound8K ----------------
print("\nDownloading UrbanSound8K...")
urban_cache = kagglehub.dataset_download(
    "rupakroy/urban-sound-8k"
)

urban_dest = os.path.join(SAVE_DIR, "UrbanSound8K")

if os.path.exists(urban_dest):
    shutil.rmtree(urban_dest)

shutil.copytree(urban_cache, urban_dest)

print(f"UrbanSound8K saved to: {urban_dest}")

print("\n✅ All datasets copied successfully!")