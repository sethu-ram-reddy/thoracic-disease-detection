# Methodology

## Problem definition

The system performs multi-label classification because a single chest radiograph can contain
multiple findings. The default experiment predicts nine findings from NIH ChestXray14:
Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, and
Consolidation.

## Leakage-safe splitting

The official NIH test list is preserved. The official training/validation set is split again at
the patient level using iterative multi-label stratification. No patient can appear in more than
one split. This is checked before training and cached in CSV manifests.

## Preprocessing

Images are converted to RGB to match ImageNet-pretrained model inputs, resized to 224 x 224,
normalized using ImageNet statistics, and augmented during training with small rotations and
horizontal flips. Validation and test images receive deterministic preprocessing only.

## Models

- DenseNet-121 uses ImageNet initialization and replaces its classifier with dropout followed by
  a nine-output linear layer.
- ViT-B/16 uses ImageNet initialization and replaces its classification head with dropout followed
  by a nine-output linear layer.

The classification head is trained for one warm-up epoch before the entire network is unfrozen.

## Optimization

The loss is binary cross-entropy with logits. A positive weight is calculated separately for every
disease from the training split to reduce the effect of class imbalance. Training uses AdamW,
cosine learning-rate decay, mixed precision on CUDA, gradient clipping, checkpointing, and early
stopping based on validation macro AUROC.

## Evaluation

Decision thresholds are selected independently for each disease on the validation split by
maximizing F1 over a fixed threshold grid. These thresholds are frozen before test evaluation.
The reported metrics include disease-level AUROC, AUPRC, precision, recall, and F1, plus macro and
micro aggregates. AUROC and AUPRC are emphasized because raw accuracy is misleading for rare
findings.

## Explainability

Grad-CAM is generated from the final DenseNet-121 convolutional normalization layer. The heatmap
shows image regions that most influenced a selected class score. It is a model-inspection aid, not
a clinical localization guarantee.

## Known limitations

- NIH ChestXray14 labels were extracted from reports and can contain noise.
- The project does not perform external validation on another hospital system.
- Demographic subgroup performance and calibration are not yet included.
- Grad-CAM can be visually persuasive without being causally faithful.
- The software is a research demonstration and is not a medical device.

