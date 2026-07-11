# Elephant Audio Classifier (YAMNet + XGBoost)

This repository contains a local training and inference pipeline for a binary elephant/non-elephant audio classifier based on a frozen YAMNet feature extractor and an XGBoost classifier. The system is designed to run efficiently on edge hardware, specifically targeting an Arduino UNO Q (Qualcomm Dragonwing QRB2210 / Cortex-A53).

## Datasets & Setup

The model uses a combination of positive (elephant) and negative (environmental) sound datasets. 

**Note on datasets:** The raw audio datasets are quite large and are deliberately excluded from this repository (see `.gitignore`). You must download them locally to the `dataset/` directory before training or evaluating.

### 1. Elephant Dataset (Positive Class)
- Placed in `dataset/data/` (organized into `train`, `test`, `validate` splits).
- Contains audio clips of elephant roars, rumbles, and trumpets.

### 2. Environmental Datasets (Negative Classes)
We use extensive negative datasets to reduce the false positive rate on ambient forest and urban noises (e.g., chainsaws, engines, weather).

**ESC-50 (Environmental Sound Classification 50)**
Downloaded via the `kagglehub` library. Contains 2,000 environmental recordings spanning 50 classes.
```python
import kagglehub
# Downloads to ~/.cache/kagglehub/datasets/mmoreaux/...
path = kagglehub.dataset_download("mmoreaux/environmental-sound-classification-50")
```
*In this project, the data was moved/symlinked to: `dataset/mmoreaux/environmental-sound-classification-50/`*

**UrbanSound8K**
Downloaded via the `kagglehub` library. Contains diverse urban noise recordings.
```python
import kagglehub
path = kagglehub.dataset_download("rupakroy/urban-sound-8k")
```
*In this project, the data was moved/symlinked to: `dataset/rupakroy/urban-sound-8k/`*

## Local Testing & Inference

You can run inference on any audio file using the provided standalone Python script. This script uses `onnxruntime` and `librosa` — PyTorch is **not** required at inference time.

```bash
# Single file inference
python test_elephant_classifier.py dataset/data/test/Trumpet/Trumpet04.wav

# Batch evaluation and profiling
python test_elephant_classifier.py --eval --profile dataset/data/test/

# Lower the threshold to 0.3 for higher recall
python test_elephant_classifier.py --eval --threshold 0.3 dataset/data/test/
```

## Arduino UNO Q Deployment

Please see [deploy_uno_q.md](deploy_uno_q.md) for detailed instructions on quantizing the model to INT8 and deploying the end-to-end Python inference daemon to the UNO Q's Cortex-A53 CPU.
