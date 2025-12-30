---
title: Oral Health AI
emoji: 🦷
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: mit
---

# 🦷 Oral Health AI - Early Detection Saves Lives

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-86.96%25-brightgreen.svg)

**A comprehensive AI-powered oral disease screening tool designed for early detection of oral cancer and other dental conditions.**

[Live Demo](https://huggingface.co/spaces/Arihant2409/oral-health-ai) • [Report Bug](https://github.com/Arihant240/oral-health-ai/issues) • [Request Feature](https://github.com/Arihant240/oral-health-ai/issues)

</div>

---

## 🎯 Problem Statement

**Oral cancer is a major health crisis in India:**
- India has the **highest rate of oral cancer globally** (1 in 10 cancers)
- **90% of cases** are linked to tobacco, gutkha, and paan consumption
- Most cases are detected at **Stage 3-4** when survival rates drop significantly
- **90%+ survival rate** if detected at Stage 1

This project aims to democratize early oral health screening using AI, making it accessible to everyone with a smartphone.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **8-Class Detection** | Detects Oral Cancer, Ulcers, Gingivitis, Caries, Calculus, Tooth Discoloration, Hypodontia, and Normal |
| 🔥 **GradCAM Visualization** | Shows WHERE the AI detected potential issues |
| 🇮🇳 **Multi-language** | Supports English and Hindi (हिंदी) |
| 📱 **Mobile-Responsive** | Works on any device with a camera |
| 📊 **Risk Assessment** | Questionnaire for tobacco/paan/smoking habits |
| 📍 **Find Dentist** | Quick link to find nearby dental clinics |
| ⚡ **Real-time Analysis** | Instant results with confidence scores |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 86.96% |
| **Training Images** | 10,860 |
| **Classes** | 8 |
| **Model Architecture** | EfficientNetB0 (Transfer Learning) |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Oral Cancer | **91%** | 77% | 83% |
| Ulcers | 100% | 97% | 98% |
| Caries | 97% | 87% | 92% |
| Tooth Discoloration | 96% | 96% | 96% |
| Gingivitis | 80% | 79% | 80% |
| Hypodontia | 69% | 94% | 80% |
| Normal Mouth | 67% | 86% | 75% |
| Calculus | 58% | 66% | 62% |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

### Training History

![Training History](results/training_history.png)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Image (224x224)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              EfficientNetB0 (Pretrained on ImageNet)         │
│                    ~4.3M Parameters                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Global Average Pooling                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│           Dense (256) → BatchNorm → Dropout (0.5)           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Dense (8) → Softmax Output                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Try the Live Demo
Visit: [Hugging Face Spaces](https://huggingface.co/spaces/your-username/oral-health-ai)

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/Arihant240/oral-health-ai.git
cd oral-health-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 3: Using Docker

```bash
docker build -t oral-health-ai .
docker run -p 8501:8501 oral-health-ai
```

---

## 📁 Project Structure

```
oral-health-ai/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── model/
│   ├── oral_disease_model.h5   # Trained TensorFlow model
│   └── class_names.json        # Class labels
│
├── results/
│   ├── confusion_matrix.png    # Model evaluation
│   └── training_history.png    # Training curves
│
├── notebooks/
│   └── training.ipynb          # Kaggle training notebook
│
└── assets/
    └── sample_images/          # Sample test images
```

---

## 🔬 Training Details

### Dataset
- **Source**: Combined from 6 Kaggle datasets
- **Total Images**: 10,860
- **Train/Val/Test Split**: 7,809 / 1,954 / 1,097

### Training Strategy
1. **Phase 1**: Frozen EfficientNetB0 base, train classification head (9 epochs)
2. **Phase 2**: Fine-tune entire model with lower learning rate (25 epochs)

### Data Augmentation
- Rotation: ±20°
- Width/Height Shift: 20%
- Shear: 15%
- Zoom: 15%
- Horizontal Flip
- Brightness: 0.8-1.2

### Class Weights
Applied to handle class imbalance (Oral Cancer had only 56 training samples)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

**This tool is for screening purposes only and is NOT a substitute for professional medical diagnosis.**

- Always consult a qualified healthcare professional for proper diagnosis
- This AI model may have limitations and can make errors
- Do not delay seeking medical advice based on results from this tool

---

## 📚 References

- [Oral Cancer Statistics - WHO](https://www.who.int/news-room/fact-sheets/detail/oral-health)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning for Medical Imaging](https://www.nature.com/articles/s41598-019-52737-x)