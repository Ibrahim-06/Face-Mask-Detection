
# 😷 Face-Mask-Detection
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-tensorflow%2C%20numpy%2C%20pandas%2C%20matplotlib%2C%20opencv--python%2C%20Pillow%2C%20scikit--learn%2C%20h5py%2C%20jupyter%2C%20seaborn%2C%20streamlit%2C%20streamlit--webrtc-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---
**Repository:** `Face-Mask-Detection`  
**Authors:** *Ibrahim Mohamed & Eslam Ahmed*

---

## 🚀 Project Overview

This project implements a **Face Mask Detection** system using **Deep Learning** (Transfer Learning with MobileNetV2). The goal is to build a robust binary classifier that distinguishes between two classes:

- `with_mask`  
- `without_mask`

Key features:
- Data preparation and splitting (train / val / test)
- Data augmentation for improved generalization
- Transfer learning with MobileNetV2 (initial training + fine-tuning)
- Model checkpointing, early stopping, and learning-rate scheduling
- Evaluation using accuracy, classification report, confusion matrix, ROC / AUC
- Explainability via Grad-CAM visualizations
- Saved model for inference and deployment

---

## 📁 Repository Structure (recommended)

```
Face-Mask-Detection/
├─ Notebook.ipynb                     
├─ dataset/                        
│  ├─ with_mask/
│  └─ without_mask/
├─ split_dataset/                  
│  ├─ train/
│  │  ├─ with_mask/
│  │  └─ without_mask/
│  ├─ val/
│  │  ├─ with_mask/
│  │  └─ without_mask/
│  └─ test/
│  │  ├─ with_mask/
│  │  └─ without_mask/
├─ models/
│  └─ Face_Mask_Model.h5           
├─ requirements.txt
├─ to_test/
│     └─ Images        
└─ README.txt    
```

> **Note:** The notebook used in this project expects the dataset at `dataset/` and will create `split_dataset/` with the structure above.

---

## 🛠️ Requirements

**Recommended Python version:** 3.8+  
Install packages with:

```bash
python -m pip install -r requirements.txt
```

Example `requirements.txt` (core packages):
```
tensorflow>=2.5
numpy
pandas
matplotlib
opencv-python
Pillow
scikit-learn
h5py
jupyter
seaborn
```

> Optional (for deployment/UI): `streamlit`, `streamlit-webrtc`

---

## 🔧 Quick Start (Notebook workflow)

1. **Clone the repo**
   ```bash
   git clone https://github.com/ibrahim-06/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```

2. **Prepare environment**
   ```bash
   python -m pip install -r requirements.txt
   ```
   
3. **Run the Notebook**
   - Open the Jupyter Notebook (e.g., `notebooks/Face_Mask_Notebook.ipynb`) and run cells in order.
   - The notebook performs:
     - Dataset split (creates `split_dataset/`)
     - Data exploration and sample visualization
     - Builds MobileNetV2-based model (224×224 input)
     - Initial training with frozen base layers
     - Fine-tuning (unfreezing last ~40 layers)
     - Model saving (`models/Face_Mask_Model.h5`)
     - Evaluation on test set (classification report, confusion matrix, ROC/AUC)
     - Grad-CAM visualizations for interpretability

---

## 🧾 Configuration & Hyperparameters (as used in the Notebook)

- `IMG_SIZE = (224, 224)`
- `BATCH_SIZE = 32`
- `EPOCHS = 12` (initial training)
- Fine-tuning: additional `6` epochs (with very small LR, e.g., `1e-5`)
- `CLASSES = ["with_mask", "without_mask"]`
- Training / Validation / Test split: `0.7 / 0.15 / 0.15`
- Callbacks:
  - `ModelCheckpoint` (monitor `val_accuracy`, save best only)
  - `EarlyStopping` (monitor `val_accuracy`, patience=4, restore best)
  - `ReduceLROnPlateau` (monitor `val_loss`, factor=0.2, patience=2)

---

## 🔍 Evaluation & Visualizations

The notebook shows how to evaluate and visualize the model results:

- **Training curves** (accuracy & loss for train vs. val)
- **Classification Report**: precision, recall, f1-score, support
- **Confusion Matrix**: counts of TP / TN / FP / FN with annotations
- **ROC Curve & AUC**: model discrimination across thresholds
- **Grad-CAM**: highlights image regions used by the model to make decisions (interpretability)

---

## 🧠 Model Architecture

- Base: `MobileNetV2` pretrained on ImageNet (`include_top=False`)
- Head:
  - `GlobalAveragePooling2D()`
  - `Dropout(0.2)`
  - `Dense(1, activation='sigmoid')` for binary output
- Training strategy:
  1. Freeze base (`base.trainable = False`) and train head (Adam, default LR)
  2. Unfreeze the last N layers (e.g., 40) and fine-tune with very low LR (`1e-5`)

---

## 🧪 Inference Example (load saved model & predict)

```python
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("models/Face_Mask_Model.h5")

# Single image inference
img = image.load_img("path/to/image.jpg", target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
x = preprocess_input(x)

prob = model.predict(x)[0,0]
label = "without_mask" if prob >= 0.5 else "with_mask"
print(f"probability: {prob:.4f} → predicted: {label}")
```

---

## 📦 Deployment Suggestions

- **Real-time**: Use OpenCV (`cv2.VideoCapture`) to capture webcam frames, preprocess them, and run model inference for real-time mask detection.
- **Web app / Demo**: Build a lightweight UI with Streamlit for demo and upload/predict flows; or expose a REST API using Flask/FastAPI for integration.
- **Edge**: Convert model to TensorFlow Lite (TFLite) for mobile or edge deployment (optimizations may be required).

---

## ✅ Tips & Best Practices

- Ensure classes are balanced or use class weighting / resampling (SMOTE is generally for tabular; for images consider augmentation or more data).
- Use stratified splitting where possible.
- Monitor validation metrics closely during fine-tuning to avoid catastrophic forgetting.
- Consider more robust augmentations or mixup/cutmix for better generalization.
- If facing overfitting: increase augmentation, add regularization, or reduce the model capacity.

---

## 🧾 Expected Outputs

When you run the notebook, you should expect:
- `split_dataset/` populated with `train/`, `val/`, `test/` folders
- Training logs with epoch-by-epoch `loss` and `accuracy`
- `models/Face_Mask_Model.h5` saved (best model)
- Evaluation artifacts: classification report, confusion matrix, ROC curve, Grad-CAM images

---

## 📝 Where to add your project links & contact

Please paste your links below in the README once available:

- **Project Video (LinkedIn post):** https://url-shortener.me/55CG
- **Contact Email:** Ibrahim.06.dev@gmail.com
- **LinkedIn Profile:** [Ibrahim Mohamed](https://www.linkedin.com/in/ibrahim-mohamed-211-)

---

## 🤝 Contribution & Credits

Contributions are welcome! If you want to improve the project, please:
1. Fork the repo
2. Create a branch (`feature/your-feature`)
3. Submit a Pull Request with a clear description of changes

Acknowledgements:
- Pretrained model: MobileNetV2 (ImageNet)
- Standard open-source libraries: TensorFlow, Keras, OpenCV, scikit-learn, etc.

---

## 📜 License

This project is released under the **MIT License** — feel free to reuse and adapt the code with attribution.

---

## 🏁 Final Notes

This Notebook & codebase was developed by **Ibrahim Mohamed** and **Eslam Ahmed**.  
If you want the README updated with specific metrics (accuracy, AUC, conf matrix values) or want the LinkedIn post & email inserted, paste them into the placeholders above and I will regenerate a final version.

Thank you for building this — great work! 🎯
