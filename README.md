# Notebook Guide

## 1) `sen12flood_proxy_labels_severity.ipynb`
**What it does:** Generates proxy labels and severity targets from SEN12-FLOOD imagery.

**How it does it:**
- Reads Sentinel-1 and Sentinel-2 scenes from the `SEN12FLOOD/` dataset.
- Creates water / flood proxy masks using spectral logic such as NDWI-style thresholding and temporal consistency rules.
- Converts those masks into severity bins and labeled examples.
- Produces manifests so downstream notebooks can reuse the same labels consistently.

**Main outputs:**
- `proxy_outputs/masks_raw/`
- `proxy_outputs/masks_temporal/`
- `proxy_outputs/proxy_labels_severity.csv`
- `proxy_outputs/severity_manifest.csv`

## 2) `flood-detection.ipynb`
**What it does:** Builds and evaluates a separate flood detection baseline.

**How it does it:**
- Preprocesses imagery and labels for a binary flood detection task.
- Uses a ResNet-based classifier to predict flooded vs. non-flooded samples.
- Trains the classifier on prepared samples and evaluates with standard classification metrics.
- Saves confusion matrices and result CSVs for comparison with the multi-task approach.

**Main outputs:**
- `proxy_outputs/resnet_confusion_matrix.csv`
- Detection-related summary plots and metrics

## 3) `flood_multitask.ipynb`
**What it does:** Trains the main multi-task flood model used in the project.

**How it does it:**
- Constructs a 7-channel input tensor from Sentinel-1 and Sentinel-2 bands.
- Trains a U-Net-style encoder-decoder with three heads:
  - segmentation head for flood mask prediction
  - detection head for flood / no-flood classification
  - severity head for ordinal severity prediction
- Uses custom losses such as Dice loss, focal/BCE variants, and ordinal regression loss.
- Saves the best checkpoint for later evaluation and analysis.

**Main outputs:**
- `checkpoints_multitask/best_multitask_unet.pt`
- `checkpoints_multitask/training_history.csv`

## 4) `cross_region_generalization.ipynb`
**What it does:** Evaluates whether the trained multi-task flood model generalizes across regions.

**How it does it:**
- Loads the multi-task U-Net checkpoint from `checkpoints_multitask/best_multitask_unet.pt`.
- Builds evaluation datasets from the SEN12-FLOOD imagery and proxy masks.
- Runs inference region-by-region and computes performance metrics.
- Produces cross-region analysis tables such as region metrics, failure analysis, and prediction summaries.

**Main outputs:**
- `proxy_outputs/cross_region_analysis/*.csv`
- Region comparison figures and summary tables


## 5) `gradcam_severity_pipeline.ipynb`
**What it does:** Creates explainability artifacts for the severity model using Grad-CAM.

**How it does it:**
- Builds a severity evaluation manifest from the project labels.
- Trains or loads a severity classifier.
- Runs Grad-CAM on selected samples to highlight image regions driving the model’s predictions.
- Generates audit tables and visual panels to inspect correct and incorrect predictions.

**Main outputs:**
- `proxy_outputs/gradcam_severity/`
- Severity audit tables and visualization panels

## 6) `vlm_vqa_benchmark.ipynb`
**What it does:** Evaluates vision-language model answers on flood-related prompts.

**How it does it:**
- Loads benchmark questions and model responses from the VLM artifacts folder.
- Compares generated answers against expected labels or categories.
- Produces benchmark summaries and confusion matrices for model comparison.

**Main outputs:**
- `vlm/benchmark_results.csv`
- `vlm/confusion_matrices/`