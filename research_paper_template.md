# Research Paper Template: CPU-Based Galaxy Classification

## Title
CPU-Efficient Galaxy Morphology Classification with Lightweight Convolutional Neural Networks

## Abstract
- Problem statement
- Dataset (Galaxy Zoo subset)
- CPU constraints
- Method summary
- Main quantitative result (accuracy, runtime)

## 1. Introduction
- Why galaxy morphology classification matters
- Why CPU-efficient ML is important for student and low-resource settings
- Research objective and contributions

## 2. Related Work
- Galaxy morphology classification approaches
- CNNs in astronomy
- Efficient deep learning on limited hardware

## 3. Dataset
- Galaxy Zoo source and curation process
- Number of images and class distribution
- Train/val/test split details

## 4. Methodology
- Preprocessing: resizing, normalization, label encoding
- Model architecture (include table with layers and parameters)
- Training setup:
  - Optimizer: Adam
  - Loss: Categorical Crossentropy
  - Epochs, batch size
  - Class weighting
  - Early stopping

## 5. Experimental Setup
- Hardware and software environment
- Runtime constraints
- Reproducibility settings (random seed)

## 6. Results
- Accuracy, precision, recall, F1
- Confusion matrix analysis
- Training and validation curves
- Runtime and inference latency

## 7. Discussion
- Error analysis
- Strengths and weaknesses
- Impact of class imbalance and mitigation effectiveness

## 8. Conclusion
- Summary of outcomes against success criteria
- Next steps (augmentation, transfer learning, richer class taxonomy)

## References
- Galaxy Zoo paper(s)
- Key machine learning and astronomy references

## Appendix
- Hyperparameter table
- Reproducibility checklist
- Command-line arguments used
