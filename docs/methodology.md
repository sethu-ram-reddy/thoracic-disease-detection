# Methodology

## Study objective

This project performs multi-label classification of nine thoracic findings and evaluates whether
a convolutional inductive bias (DenseNet-121) or a patch-based transformer (ViT-B/16) is more
effective under an identical transfer-learning budget on NIH ChestXray14.

## Dataset and target definition

The audited dataset contains 112,120 radiographs. A study may contain more than one finding, so
each label is encoded independently. The evaluated targets are Atelectasis, Cardiomegaly, Effusion,
Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, and Consolidation.

The project intentionally does not call the output a diagnosis. ChestXray14 labels were extracted
from reports and may be missing, incorrect, or clinically ambiguous.

## Leakage-safe splitting

The official NIH test list is preserved. The official training/validation list is divided at the
patient level using iterative multi-label stratification. Patient-level targets are calculated by
taking the maximum label value across a patient's images before stratification.

| Split | Images | Patients |
|---|---:|---:|
| Train | 73,254 | 23,806 |
| Validation | 13,270 | 4,202 |
| Test | 25,596 | 2,797 |

The data module verifies that no patient identifier occurs in two partitions. Split manifests are
cached with the experiment output so training, threshold selection, and evaluation use exactly the
same observations.

## Image preprocessing

Images are decoded as RGB because both backbones use ImageNet initialization. All inputs are
resized to 224 × 224 and normalized with ImageNet mean and standard deviation.

Training images receive a random horizontal flip and a random rotation within ±7 degrees.
Validation and test preprocessing is deterministic. No augmentation is applied while selecting
thresholds or reporting final metrics.

## Architectures

### DenseNet-121

The ImageNet classifier is replaced by dropout and a nine-output linear layer. Dense connectivity
supports feature reuse and is well suited to local texture and opacity patterns in radiographs.

### ViT-B/16

The ImageNet classification head is replaced by dropout and a nine-output linear layer. The model
represents the image as 16 × 16 patches and learns global interactions through self-attention.

For both models, only the classification head is trainable during epoch one. The full backbone is
unfrozen from epoch two onward.

## Optimization

The objective is `BCEWithLogitsLoss`. For every disease, the positive weight is calculated as the
number of negative training examples divided by the number of positive training examples. This
reduces domination by the negative class without resampling the official image distribution.

Both experiments use AdamW, cosine learning-rate decay, gradient clipping at 1.0, mixed precision,
best/latest checkpointing, and early stopping on validation macro AUROC.

| Setting | DenseNet-121 | ViT-B/16 |
|---|---:|---:|
| Epochs | 8 | 8 |
| Batch size | 64 | 48 |
| Initial learning rate | 3e-4 | 1e-4 |
| Weight decay | 1e-4 | 1e-4 |
| Warm-up head-only epochs | 1 | 1 |

The completed runs used PyTorch 2.11 with CUDA 12.8 on an NVIDIA A100-SXM4 80 GB GPU.

## Threshold selection and evaluation

The models emit nine independent probabilities. A single threshold of 0.5 is not assumed to be
appropriate across findings with different prevalence and score distributions.

For each disease, a threshold is selected on validation predictions by maximizing F1 over values
from 0.05 through 0.95 in increments of 0.05. Thresholds are then frozen. The test set is processed
once to produce disease-level AUROC, AUPRC, precision, recall, and F1, plus macro and micro
aggregates.

AUROC measures ranking across thresholds. AUPRC is emphasized alongside AUROC because it is more
sensitive to performance on rare positive findings. F1 describes the selected operating point but
does not express calibration or clinical utility.

## Explainability

Grad-CAM is generated for DenseNet-121 from
`features.denseblock4.denselayer16.conv2`, the final convolution in the last dense block. Gradients
of a selected class logit are spatially averaged and used to weight activation maps. The resulting
map is rectified, resized to input resolution, normalized, and overlaid on the radiograph.

The final feature map is spatially coarse, so these visualizations indicate broad influential
regions. They are not pixel-accurate lesion boundaries and must not be interpreted as causal or
clinically validated localization.

## Results interpretation

DenseNet-121 achieved 0.7826 macro AUROC and 0.2997 macro AUPRC. ViT-B/16 achieved 0.7544 and
0.2638 respectively. DenseNet also produced higher AUROC for every evaluated disease.

The controlled result supports DenseNet-121 for this dataset and budget. It does not establish
that convolutional models universally outperform transformers. More extensive transformer
pretraining, resolution, augmentation, regularization, or optimization could change the result.

## Limitations and responsible use

- The labels are weak report-derived annotations rather than adjudicated diagnoses.
- Nine of fourteen ChestXray14 findings are evaluated.
- No external hospital dataset is used for validation.
- There is no subgroup fairness analysis or confidence calibration study.
- Images are resized to 224 × 224, which may suppress subtle findings.
- The experiment is not optimized for clinical sensitivity or specificity requirements.
- Grad-CAM can appear plausible even when a model relies on confounding information.

The repository is a machine-learning engineering and research demonstration. It is not intended
for patient care, triage, or medical decision support.
