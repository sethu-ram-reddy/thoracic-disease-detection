# Interpretable Thoracic Disease Classification

An end-to-end PyTorch project for multi-label classification of nine thoracic findings from chest
X-rays. It benchmarks a convolutional architecture (DenseNet-121) against a transformer
(ViT-B/16), evaluates disease-level performance, and uses Grad-CAM to inspect the image regions
driving DenseNet predictions.

> Research software only. This repository is not intended for diagnosis or clinical use.

## Project status

The reproducible training and evaluation pipeline is implemented. Full NIH ChestXray14 experiments
must be executed before publishing final performance claims. The repository intentionally contains
no invented or estimated metrics.

## What this project demonstrates

- Multi-label learning on 112,000+ chest radiographs
- Patient-disjoint training, validation, and test splits
- Class-imbalance handling with disease-specific positive weights
- DenseNet-121 and ViT-B/16 transfer learning
- Mixed-precision training, checkpointing, and early stopping
- Validation-only threshold selection
- Disease-level AUROC, AUPRC, precision, recall, and F1
- Grad-CAM explainability for DenseNet-121
- Reproducible Colab execution with outputs persisted to Google Drive

## Target findings

The default configuration predicts Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass,
Nodule, Pneumonia, Pneumothorax, and Consolidation. The class list is configuration-driven and can
be expanded to all 14 NIH findings.

## Repository structure

```text
configs/                 DenseNet-121 and ViT experiment definitions
docs/                    Methodology and limitations
notebooks/               Colab orchestration notebook
scripts/                 Training, evaluation, comparison, and Grad-CAM entry points
src/thoracic_ai/         Reusable data, model, metric, and training modules
tests/                   Configuration and metric tests
results/                 Final tracked figures and tables after experiments
```

## Experimental design

The official NIH test split is preserved. The official training/validation list is divided at the
patient level using iterative multi-label stratification. Integrity checks stop execution if a
patient appears in more than one split.

Both models start from ImageNet weights. Their classification heads are trained for one warm-up
epoch, followed by full fine-tuning. Binary cross-entropy with disease-specific positive weights
addresses label imbalance. Early stopping monitors validation macro AUROC.

Thresholds are selected per disease using only the validation set. The untouched test set is then
used once for final reporting. See [the complete methodology](docs/methodology.md).

## Quick start

The full dataset is approximately 42 GB, so Google Colab with GPU acceleration is the recommended
environment. The notebook stages the images from Google Drive onto Colab's local disk to avoid the
I/O bottleneck caused by training directly from a mounted Drive.

```bash
pip install -r requirements-colab.txt
pip install -e .
```

The NIH directory must contain `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt`, and the
12 image archives after extraction.

```bash
python scripts/audit_dataset.py --data-root /content/data/nih-chest-xray
```

### Train and evaluate DenseNet-121

```bash
python scripts/train.py \
  --config configs/densenet121.yaml \
  --data-root /content/data/nih-chest-xray

python scripts/evaluate.py \
  --config configs/densenet121.yaml \
  --data-root /content/data/nih-chest-xray

python scripts/generate_gradcam.py \
  --config configs/densenet121.yaml \
  --data-root /content/data/nih-chest-xray
```

### Train and evaluate ViT-B/16

```bash
python scripts/train.py \
  --config configs/vit_b16.yaml \
  --data-root /content/data/nih-chest-xray

python scripts/evaluate.py \
  --config configs/vit_b16.yaml \
  --data-root /content/data/nih-chest-xray
```

### Compare the models

```bash
python scripts/compare_models.py \
  --densenet-dir /content/drive/MyDrive/Thoracic_Disease_Detection/outputs/densenet121 \
  --vit-dir /content/drive/MyDrive/Thoracic_Disease_Detection/outputs/vit_b16 \
  --output-dir results
```

## Outputs

Each experiment writes the best and latest checkpoints, resolved configuration, patient-safe split
manifests, training history, validation-selected thresholds, test predictions, metric tables, and
publication-ready figures. Checkpoints and raw predictions remain untracked because of their size;
final comparison tables and representative figures belong in `results/`.

## Results

This section will be populated from the generated artifacts after both full experiments finish.
No result should be added to the README, resume, or portfolio until it can be traced to the saved
test evaluation.

## Author

Sethu Ram Reddy Lankala — M.S. Computer Engineering, Machine Learning & Intelligent Systems,
The George Washington University.

## License

Released under the MIT License. The NIH ChestXray14 dataset is distributed separately under its
own terms and is not included in this repository.

