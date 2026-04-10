# 🌌 GalaxyNet V25: Technical Deep Dive

Welcome to the official technical guide for GalaxyNet V25. This document serves as the primary source of truth for the **Two-Stage Cascade Pipeline** architecture, designed to push the boundaries of galaxy morphology classification.

---

## 🏛️ System Architecture

GalaxyNet V25 transitions from a flat ensemble to a hierarchical cascade. This design is motivated by the "easy-to-hard" nature of galaxy classification: Elliptical galaxies are highly distinct, while the boundary between Spiral and Irregular galaxies is notoriously fuzzy.

### Cascade Decision Flow

```mermaid
graph TD
    A[🌌 Galaxy Image 224x224] --> B{Stage 1: EN-V2B1}
    B -- "P(Elliptical) ≥ 0.469" --> C[⭕ Predict: Elliptical]
    B -- "P(Elliptical) < 0.469" --> D[🔎 Stage 2: EN-V2B2 + B1 Ensemble]
    
    D -- "P(Spiral) ≥ Calibrated T2" --> E[🌀 Predict: Spiral]
    D -- "P(Spiral) < Calibrated T2" --> F[✨ Predict: Irregular]

    style C fill:#2196F3,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#4CAF50,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#FF5722,stroke:#333,stroke-width:2px,color:#fff
```

> [!TIP]
> **Why this works:** By isolating Ellipticals in Stage 1, we prevent "clean" smooth galaxies from regularizing the Stage 2 models, allowing Stage 2 to focus exclusively on distinctive feature learning (arms vs. asymmetries).

---

## 🚀 Training Pipeline

We employ a **3-Phase Resolution Curriculum** combined with **Stochastic Weight Averaging (SWA)** to ensure maximum generalization.

### The Curriculum Flowchart

```mermaid
sequenceDiagram
    participant P1 as Phase 1: Warmup (128px)
    participant P2 as Phase 2: Mid-Tune (192px)
    participant P3 as Phase 3: Fine-Tune (224px)
    participant SWA as Step 4: SWA Averaging

    Note over P1: Frozen Backbone<br/>High LR (1e-3)
    P1->>P2: Unfreeze last 50 layers
    Note over P2: Mid LR (3e-5)<br/>Increase Resolution
    P2->>P3: Unfreeze last 80 layers
    Note over P3: Low LR (8e-6)<br/>SGDR Scheduler
    P3->>SWA: 5 Epochs Cyclical LR
    Note over SWA: Weight Averaging<br/>BN Update Pass
```

### Advanced Training Components

| Feature | Implementation | Benefit |
| :--- | :--- | :--- |
| **LLRD** | Layer-wise Learning Rate Decay | Prevents catastrophic forgetting in early backbone layers. |
| **SGDR** | Cosine Annealing with Restarts | Helps the optimizer escape local minima at high resolutions. |
| **SWA** | Stochastic Weight Averaging | Produces a model in a wider minimum, increasing test robustness by ~1%. |
| **Focal Loss** | Binary Focal (Stage 1: γ=1.5 / Stage 2: γ=2.5) | Forces the model to focus on hard, misclassified examples. |

---

## 🧪 Data Engineering & Augmentation

The "Irregular" class is our hardest target. We use **within-class Mixup** and **multi-hole Cutout** to force the model to learn distributed, non-occlusion-sensitive features.

### Augmentation Strategy

> [!NOTE]
> **Mixup Alpha (0.4):** Applied at the batch level for irregular samples to create synthetic training points.
> **Cutout (3 Holes):** For irregulars, we use up to 20% image size holes to ensure the model doesn't rely on a single clump of pixels.

```mermaid
graph LR
    I[Input] --> R[Rotate/Flip]
    R --> C[Cutout/Mixup]
    C --> B[Blur/Contrast]
    B --> O[224px Output]
```

---

## 📊 Inference & Calibration

Thresholds are not static. V25 introduces a **2D Grid Search** on the validation set to find the optimal boundary between stages.

### 16-TTA (Test Time Augmentation)
To ensure stable predictions, we average results across 16 transformations:
1.  **4 Rotations** (0°, 90°, 180°, 270°)
2.  **2 Flips** (None, Horizontal)
3.  **2 Center Crops** (0.75, 0.85)

### Error Decompositon
We track errors in three categories:
-   **Stage 1 FN:** Missed Ellipticals (Leaking to Stage 2).
-   **Stage 1 FP:** Non-Ellipticals blocked (Forced to Elliptical).
-   **Stage 2 Confusion:** Spiral/Irregular misclassifications.

---

## 📉 Data Generation Flow

The journey from consensus votes to cleaned morphological labels is governed by our V25 threshold logic.

```mermaid
graph TD
    S[CSV: solutions_rev1.csv] --> P[Process: make_labels.py]
    P --> C1{Class 1.1 ≥ 0.469?}
    C1 -- Yes --> E[⭕ Elliptical]
    C1 -- No --> C6{Class 6.1 ≥ 0.5?}
    
    C6 -- Yes --> M{Margin ≥ 0.15?}
    M -- Yes --> I[✨ Irregular]
    M -- No --> D[Discard]
    
    C6 -- No --> C4{Class 4.1 ≥ 0.45?}
    C4 -- Yes --> G[🌀 Spiral]
    C4 -- No --> D
```

---

## 🔍 Inference Detail: TTA-16

During evaluation, we don't just take a single snapshot. We use Test-Time Augmentation to stabilize the sigmoid outputs.

```mermaid
flowchart LR
    Img[Single Image] --> T1[4x Rotations]
    Img --> T2[2x Flips]
    Img --> T3[2x Crops]
    
    T1 & T2 & T3 --> Combined[16x Transformed Images]
    Combined --> Model[Cascade Model]
    Model --> Preds[16x Sigmoid Probs]
    Preds --> Avg[Averaged Probability]
    Avg --> Thres[Final Class Assignment]
```

---

## 🛠️ Contributor Quickstart

### Project Layout
- `config.py`: Central hub for all hyperparameters. Change resolutions and LRs here.
- `dataset.py`: The heart of data loading and label generation.
- `train.py`: The orchestrator. Isolated subprocesses prevent VRAM OOMs.
- `evaluate.py`: Detailed TTA-16 performance reporting.

### How to Extend
1. **Backbone Architecture**: Modify `stage1_architecture` in `config.py` to test different EfficientNet or ConvNeXt variants.
2. **Loss Weights**: Adjust `pos_weight` calculation in `train.py` if class imbalance shifts.
3. **Threshold Calibration**: The calibration logic in `evaluate.py` can be updated to prioritize specific metrics like Irregular F1.

---

## 📚 Glossary

- **Cascade Architecture**: A multi-stage classification setup where samples are filtered through specialized models.
- **SWA (Stochastic Weight Averaging)**: A technique that averages model weights across multiple training points to achieve a smoother loss landscape and better generalization.
- **TTA (Test-Time Augmentation)**: Running multiple augmented versions of a single image through the model and averaging the results for a more stable prediction.
- **SGDR (Stochastic Gradient Descent with Restarts)**: A learning rate schedule that periodically "restarts" to high LR to escape local minima.
- **Focal Loss**: A loss function that applies a modulating factor to the cross-entropy loss, down-weighting well-classified examples and focusing on hard ones.

## ❓ Frequently Asked Questions

**Q: Why use EfficientNetV2 specifically?**
A: EfficientNetV2-B1 and B2 offer a superior balance between parameter efficiency and accuracy compared to standard ResNet or ConvNeXt variants for this specific resolution (224px).

**Q: Can I run this without a GPU?**
A: Training is computationally intensive and realistically requires a GPU (P100 or better). However, inference using the `.keras` files can run on a CPU.

**Q: What happens if Stage 1 is wrong?**
A: If Stage 1 misclassifies a Non-Elliptical as an Elliptical, it is "blocked" from Stage 2. We use a high filtering confidence (0.85) to minimize this "leakage."

**Q: How do I change the thresholds?**
A: Thresholding is handled dynamically in `evaluate.py` via grid search on the validation set, but the "loose" filter for training data is adjusted in `config.py` under `stage1_filter_confidence`.

---

> [!IMPORTANT]
> **Acceptance Criteria:** A production-ready run must achieve >95% accuracy and an Irregular F1 > 0.80. If these fail, the `auto-fallback` logic in `train.py` will automatically trigger with relaxed thresholds.
