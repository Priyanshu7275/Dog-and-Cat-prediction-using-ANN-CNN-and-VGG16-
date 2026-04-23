# Cats vs Dogs — Deep Learning Model Comparison

A deep learning project that classifies images of **cats and dogs** using three progressively more powerful approaches:

1. **Artificial Neural Network (ANN)** — baseline fully-connected network
2. **Convolutional Neural Network (CNN)** — custom architecture built from scratch
3. **VGG16 (Transfer Learning)** — pretrained ImageNet model used as feature extractor

The goal of the project is to **compare how different architectures handle image classification** and demonstrate why CNNs and transfer learning outperform fully-connected networks on visual data.

---

## Project Structure

```
├── cats_and_dogs_ann_model.ipynb        # ANN baseline model
├── cats_and_dogs__cnn_model_.ipynb      # Custom CNN model
├── cats_and_dogs_using_vgg16.ipynb      # Transfer learning with VGG16
├── train/
│   ├── cat/
│   └── dog/
└── val/
    ├── cat/
    └── dog/
```

---

## Dataset

- **Classes:** `cat` and `dog` (binary classification)
- **Image size:** Resized to **128×128×3**
- **Split:** Separate `train/` and `val/` folders, each with `cat/` and `dog/` subfolders
- **Loading:** Images are read with OpenCV (`cv2.imread`), resized, and normalized to `[0, 1]`

---

## Data Preprocessing & Augmentation

All three models use a consistent preprocessing pipeline for fair comparison:

- Resize images to **128×128**
- Normalize pixel values (`/255.0`)
- Apply **data augmentation** on training data only (`ImageDataGenerator`):
  - Rotation (±20–25°)
  - Width/Height shift (±10–20%)
  - Zoom (±20%)
  - Horizontal flip
  - Shear
- Validation data uses a plain generator (no augmentation) for reliable evaluation

---

## Model 1 — ANN (Baseline)

A fully-connected network used as a **baseline** to show how a plain ANN performs on image data when it has no understanding of spatial structure.

**Architecture:**
```
Flatten (128×128×3 → 49,152)
→ Dense(512, relu) + BatchNorm + Dropout(0.3)
→ Dense(256, relu) + Dropout(0.3)
→ Dense(128, relu)
→ Dense(64, relu)
→ Dense(1, sigmoid)
```

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 20
- **Total params:** ~25.3M

### ANN Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **65.71%** |
| Validation Loss | 0.7989 |

**Confusion Matrix:**
```
              Predicted Cat   Predicted Dog
Actual Cat         2              22
Actual Dog         2              44
```

**Classification Report:**
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Cat   | 0.50      | 0.08   | 0.14     |
| Dog   | 0.67      | 0.96   | 0.79     |

**Observation:** The ANN mostly predicts "dog" for every image, giving it a decent accuracy but very poor recall for cats. This confirms the hypothesis — **flattening images destroys spatial structure**, and a fully-connected network cannot learn meaningful visual features from raw pixel values. This is exactly why CNNs exist.

---

## Model 2 — CNN (Custom)

A custom Convolutional Neural Network designed specifically for image classification. Uses **4 convolutional blocks** followed by dense layers with regularization.

**Architecture:**
```
Conv2D(32)  → BN → MaxPool → Dropout(0.25)
Conv2D(64)  → BN → MaxPool → Dropout(0.25)
Conv2D(128) → BN → MaxPool → Dropout(0.30)
Conv2D(256) → BN → MaxPool → Dropout(0.50)
Flatten
→ Dense(512) → BN → Dropout(0.5)
→ Dense(256) → BN → Dropout(0.5)
→ Dense(128) → BN → Dropout(0.5)
→ Dense(1, sigmoid)
```

**Techniques used:**
- **Class weighting** to handle class imbalance
- **EarlyStopping** (patience = 5)
- **ReduceLROnPlateau** for adaptive learning rate
- **BatchNormalization** + **Dropout** for regularization

- **Optimizer:** Adam
- **Loss:** Binary Crossentropy
- **Epochs:** 25

---

## Model 3 — VGG16 (Transfer Learning)

Uses the **pretrained VGG16** model (trained on ImageNet) as a feature extractor, with custom dense layers on top.

**Architecture:**
```
VGG16 base (frozen, no top)
→ Conv2D(512, relu, L2 regularization)
→ GlobalAveragePooling2D
→ Dense(512) → Dropout(0.5)
→ Dense(256) → Dropout(0.4)
→ Dense(128) → Dropout(0.3)
→ Dense(1, sigmoid)
```

- **Base model:** VGG16 (ImageNet weights, `include_top=False`)
- **Frozen layers:** All VGG16 layers (only head is trained)
- **L2 regularization** on top convolutional layer
- **ReduceLROnPlateau** callback
- **Optimizer:** Adam
- **Epochs:** 50

**Why VGG16?** Transfer learning uses features already learned from millions of ImageNet images (edges, textures, shapes) — these transfer well to cat/dog classification.

---

## Tech Stack

- **Python 3**
- **TensorFlow / Keras** — model building and training
- **OpenCV** — image loading and resizing
- **NumPy, Pandas** — data handling
- **scikit-learn** — class weighting, confusion matrix, classification report
- **Matplotlib** — visualization

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd cats-vs-dogs
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow opencv-python numpy pandas scikit-learn matplotlib
   ```

3. Organize your dataset:
   ```
   train/cat/   train/dog/
   val/cat/     val/dog/
   ```

4. Run any notebook in Jupyter:
   ```bash
   jupyter notebook cats_and_dogs_ann_model.ipynb
   ```

---

## Key Learnings

- **ANN is unsuitable for images** — confirmed by the 65.7% accuracy collapsing to 8% recall on the minority class. Flattening destroys spatial information.
- **CNNs learn spatial hierarchies** through convolution and pooling layers — they see local patterns like edges and textures, building up to complex shapes.
- **Transfer learning saves time and improves accuracy** — especially powerful when training data is limited.
- **Accuracy alone is misleading** on imbalanced datasets — confusion matrix and per-class recall tell the real story.
- **Regularization matters** — BatchNorm, Dropout, EarlyStopping, and data augmentation all help generalization.

---

## Future Improvements

- **Expand the dataset** — use the full Kaggle Dogs vs Cats dataset (25,000 images) for more reliable results
- **Fine-tune VGG16 layers** instead of keeping all frozen
- **Try other pretrained backbones:** ResNet50, EfficientNet, MobileNetV2
- **Use VGG16's native preprocessing** (`vgg16.preprocess_input`) instead of `/255`
- **Increase input size to 224×224** (VGG16's native input)
- **Save trained models** as `.h5` files
- **Build a Streamlit demo** for live predictions

---

## Author

Built as a deep learning practice project to explore the progression from basic ANNs to CNNs to transfer learning on a standard image classification task.
