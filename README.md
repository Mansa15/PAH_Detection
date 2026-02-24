# 🫁 Detection of Abnormalities in the Pulmonary Artery (PAH Detection)

> A deep learning system for early detection of **Pulmonary Artery Hypertension (PAH)** from chest X-rays and CT scans using Transfer Learning (EfficientNetB0).

---

## 👥 Team

| Name | Roll No |
|------|---------|
| Ved Dharmendra Patel | 22WUO102223 |
| Sai Pratyush Reddy | 22WUO102206 |
| Mansa Thallapalli | 22WUO102216 |
| Thanmayee Bethireddy | 22WUO102217 |

**Supervised by:** Dr. Bhargav Prajwal Pathri, Assistant Professor, School of Technology

---

## 📌 Objectives

1. Train a model to detect early-stage Pulmonary Artery Hypertension
2. Analyze PAH conditions and predict future possibilities for any PAH patient

---

## 🏗 Project Structure

```
pah-detection/
├── data/
│   └── images/
│       ├── normal/         ← Healthy chest X-rays / CT scans
│       └── pah/            ← PAH-positive images
├── models/
│   ├── best_model.h5       ← Best checkpoint (auto-saved during training)
│   ├── pah_model_final.h5  ← Final trained model
│   ├── class_indices.json  ← Class label mapping
│   └── training_curves.png ← Loss/accuracy plots
├── src/
│   ├── train.py            ← Model training script
│   ├── predict.py          ← CLI inference script
│   ├── data_prep.py        ← Data preparation utilities
│   └── app.py              ← Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙ Setup

### 1. Clone / Download the project
```bash
git clone https://github.com/your-username/pah-detection.git
cd pah-detection
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Preparing Your Dataset

Organize your images into this structure:

```
data/images/
    normal/   ← X-rays with no PAH (e.g., 500 images)
    pah/      ← X-rays showing PAH  (e.g., 500 images)
```

### Recommended Public Datasets
- **NIH Chest X-ray Dataset** — [https://nihcc.app.box.com/v/ChestXray-NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **ChestX-ray14** — [https://stanfordmlgroup.github.io/](https://stanfordmlgroup.github.io/)
- **MIMIC-CXR** — [https://physionet.org/content/mimic-cxr/](https://physionet.org/content/mimic-cxr/)

> Filter images with `Pulmonary Hypertension` labels as the PAH class and `No Finding` as the Normal class.

### Validate your dataset
```bash
python src/data_prep.py --validate --data-dir data/images
```

### Run EDA
```bash
python src/data_prep.py --eda --data-dir data/images
```

### Convert DICOM to JPEG (if needed)
```bash
pip install pydicom
python src/data_prep.py --convert-dicom path/to/dicoms --dicom-out data/images/pah
```

---

## 🧠 Model Architecture

```
Input (224×224×3)
     ↓
EfficientNetB0 (ImageNet pretrained, backbone)
     ↓
GlobalAveragePooling2D
     ↓
BatchNormalization
     ↓
Dense(256, ReLU) → Dropout(0.4)
     ↓
Dense(128, ReLU) → Dropout(0.3)
     ↓
Dense(1, Sigmoid)   ← Binary output: Normal / PAH
```

**Training Strategy:**
- Phase 1: Freeze backbone → train top layers (20 epochs)
- Phase 2: Unfreeze last 30 backbone layers → fine-tune (up to 50 epochs)
- Class weights computed automatically for imbalanced datasets

---

## 🚀 Training

```bash
python src/train.py
```

**What happens:**
1. Images are loaded with augmentation (rotation, flip, zoom)
2. Class weights are computed to handle imbalanced data
3. EfficientNetB0 backbone is fine-tuned
4. Best model is saved to `models/best_model.h5`
5. Training curves are saved to `models/training_curves.png`

**Expected metrics (good training):**
- Accuracy > 85%
- AUC > 0.90
- Training time: ~30–90 min on GPU, longer on CPU

---

## 🔍 Inference (CLI)

### Single image
```bash
python src/predict.py --image path/to/xray.jpg
```

### Batch (folder of images)
```bash
python src/predict.py --folder path/to/test_images/
```

**Output example:**
```
file                 : patient001.jpg
probability_pah      : 0.8342
prediction           : pah
confidence           : 83.42%
risk_level           : High
```

Results are also saved to `predictions.csv`.

---

## 🌐 Web Application (Streamlit)

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Features:**
- Upload chest X-ray / CT scan image
- Real-time PAH probability prediction
- Risk level indicator (Low / Moderate / High / Very High)
- **Grad-CAM explainability** — highlights regions the model focused on
- Adjustable decision threshold via sidebar

---

## 📊 Methodology

```
Raw Data
   ↓
Pre-Processing
   • Resize to 224×224
   • Normalize pixel values [0,1]
   • Data augmentation (rotation, flip, zoom, shift)
   ↓
Class Weights Calculation
   • compute_class_weight("balanced") from sklearn
   ↓
Model Definition & Compilation
   • EfficientNetB0 + custom top layers
   • Binary crossentropy loss
   • Adam optimizer
   ↓
Model Training & Saving
   • Phase 1: Top layers only
   • Phase 2: Fine-tuning
   • Best checkpoint saved automatically
   ↓
Testing & Evaluation
   • Accuracy, AUC, Precision, Recall
   • Classification Report
   • Confusion Matrix
```

---

## 📈 Future Scope

- Fine-tune with larger, more diverse datasets for higher accuracy
- Integrate with PACS (hospital imaging systems)
- Add multi-class classification (severity levels: mild / moderate / severe)
- Mobile deployment for point-of-care screening
- Longitudinal analysis: track PAH progression over time

---

## 📚 References

1. Frank et al. (1993). Detection and quantification of pulmonary artery hypertension with MR imaging. *AJR American Journal of Roentgenology*, 161(1), 27–31.
2. Santos-Gomes et al. (2022). An overview of circulating pulmonary arterial hypertension biomarkers. *Frontiers in Cardiovascular Medicine*, 9, 924873.
3. Enhanced Pulmonary Embolism Detection in CT Angiography Using Spectral Imaging and Deep Learning. *SN Computer Science*, 5(2), Article 3352.
4. Akilandeswaria et al. (2021). Detecting pulmonary embolism using deep neural networks. *Int J Perform Eng*, 17(3), 322–332.

---

## ⚠ Disclaimer

This project is developed for **academic and research purposes only**. It is **NOT** intended for clinical use or as a substitute for professional medical diagnosis. Always consult a qualified physician.

---

## 📄 License

MIT License — free to use for academic purposes with attribution.
