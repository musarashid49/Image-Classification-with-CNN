# IEEE Report Outline
## Pakistani Politician Image Classification using CNNs

> Use this as your Overleaf section guide. Each heading below = one LaTeX \section or \subsection.

---

### I. Introduction
- Problem statement: multi-class facial recognition of 16 Pakistani public figures
- Motivation and real-world relevance
- Summary of approach (pretrained CNNs, transfer learning)
- Paper organisation

### II. Related Work
- CNN-based face recognition (FaceNet, ArcFace overview — cite only)
- Transfer learning for small datasets
- Prior Pakistani political face recognition work (if any)

### III. Dataset
#### A. Collection Methodology
- Sources: Google Images, Wikipedia, news sites
- Tools used (manual download / scraper)
- Per-class image counts table

#### B. Dataset Statistics
- Table: class name | train | val | test | total
- Distribution figure: `results/plots/dataset_distribution.png`

#### C. Preprocessing
- Image resizing strategy (224×224)
- ImageNet normalisation (mean, std)

### IV. Data Augmentation
- Justification: small dataset → need augmentation to reduce overfitting
- Table of transforms applied (training only):
  | Transform | Parameters |
  |---|---|
  | RandomResizedCrop | scale (0.85, 1.0) |
  | RandomHorizontalFlip | p=0.5 |
  | RandomRotation | ±20° |
  | ColorJitter | brightness=0.3, contrast=0.2 |
- Explicitly state: val and test use only Resize + CenterCrop + Normalize

### V. CNN Architectures
#### A. ResNet-50
- Architecture overview (skip connections, bottleneck blocks)
- Pretrained weights: ImageNet1K-V2
- Custom head: Dropout(0.4) → Linear(2048, 16)
- Parameter count

#### B. EfficientNet-B0
- Architecture overview (compound scaling, MBConv blocks)
- Pretrained weights: ImageNet1K-V1
- Custom head: Dropout(0.4) → Linear(1280, 16)
- Parameter count

### VI. Training Strategy
- Optimiser: Adam (lr=1e-4, weight_decay=1e-4)
- Scheduler: ReduceLROnPlateau (patience=4, factor=0.5)
- Early stopping: patience=8 epochs
- Loss function: CrossEntropyLoss
- Batch size: 32
- Hardware: Google Colab A100 GPU
- Epochs: up to 30 (early stopping may terminate earlier)

### VII. Results
#### A. Overall Accuracy
- Table: Model | Test Accuracy | Macro F1

#### B. Per-Class Metrics
- Table from `results/metrics/per_class_metrics.csv`

#### C. Confusion Matrices
- Figure: `results/plots/resnet50_confusion_matrix.png`
- Figure: `results/plots/efficientnet_b0_confusion_matrix.png`

#### D. Training Curves
- Figure: `results/plots/resnet50_training_curves.png`
- Figure: `results/plots/efficientnet_b0_training_curves.png`

#### E. Model Comparison
- Figure: `results/plots/model_comparison.png`

### VIII. Misclassification Analysis
- Top-5 misclassified samples per model
- Figure: `results/plots/resnet50_misclassified.png`
- Common error patterns (visual similarity between classes)
- Suggestions for improvement

### IX. Challenges Faced
- Dataset collection difficulties
- Class imbalance
- Visual similarity between certain politicians
- Colab session limits

### X. Conclusion
- Summary of findings
- Best performing model and justification
- Future work: larger dataset, face detection preprocessing, ensemble methods

### References
- He et al. (2016) — Deep Residual Learning (ResNet)
- Tan & Le (2019) — EfficientNet
- Deng et al. (2009) — ImageNet
- Any other cited papers
