# 🦷 OralSense AI — Intelligent Oral Lesion Classification

### Confidence-Aware Adaptive Oral Image Classification Using Deep Learning and Explainable AI

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-FF4B4B?logo=streamlit\&logoColor=white)](https://oralsense-ai.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?logo=huggingface)
![Status](https://img.shields.io/badge/Project-Research%20Prototype-green)

**OralSense AI** is an AI-powered oral image classification system integrating deep learning, machine learning, adaptive decision-making, and Grad-CAM visualization through an interactive Streamlit web application.

The system combines **ConvNeXt-Tiny**, **DINOv2-SVM**, and a learned adaptive decision policy to classify oral images while accounting for prediction uncertainty.

### 🌐 Live Application

**[🚀 Launch OralSense AI](https://oralsense-ai.streamlit.app)**

> **Note:** OralSense AI is an academic research prototype and is not intended for clinical diagnosis.

---

## 📌 Project Overview

Oral conditions can exhibit similar visual characteristics, making automated image classification challenging.

Traditional deep learning classifiers typically rely on a single model to produce predictions, even when confidence is low.

OralSense AI explores a confidence-aware approach that combines two complementary classification models with an adaptive decision mechanism.

The system first analyzes an uploaded image using ConvNeXt-Tiny. When the primary prediction is uncertain, DINOv2-SVM provides a secondary assessment.

A learned decision policy determines whether to accept the specialist prediction or recommend professional review.

Grad-CAM provides visual explanations of the primary model's predictions.

---

## ✨ Key Features

* **Oral Image Classification:** Classifies images across six oral-condition categories.
* **ConvNeXt-Tiny:** Primary deep learning classifier.
* **DINOv2-SVM:** Secondary classification model for uncertain predictions.
* **Confidence-Aware Inference:** Identifies predictions requiring additional assessment.
* **Adaptive Decision Policy:** Uses a learned gate to evaluate specialist predictions.
* **Referral Mechanism:** Avoids automatically assigning a disease label when predefined acceptance conditions are not satisfied.
* **Grad-CAM Visualization:** Highlights image regions influencing the primary model's prediction.
* **Interactive Web Application:** Provides image uploads, classification results, confidence scores, and explanations.
* **Cloud Deployment:** Hosted using Streamlit Community Cloud.
* **Automated Model Setup:** Downloads trained model artifacts from Hugging Face.

---

## 🧠 System Architecture

```text
                  Uploaded Oral Image
                          |
                          v
                 Image Preprocessing
                          |
                          v
                    ConvNeXt-Tiny
                          |
                          v
                   Confidence Check
                          |
              +-----------+-----------+
              |                       |
          Confident                Uncertain
              |                       |
              v                       v
       Primary Prediction        DINOv2 + SVM
                                      |
                                      v
                              Learned Adaptive Gate
                                      |
                              +-------+-------+
                              |               |
                              v               v
                      Accept Specialist    Refer for
                         Prediction       Expert Review
                             
                  Grad-CAM Visualization
                 (Primary Model Explanation)
```

### Working Principle

1. The user uploads an oral image through the Streamlit application.
2. The image is preprocessed using the model's predefined input transformations.
3. ConvNeXt-Tiny generates the initial class prediction and confidence score.
4. The system evaluates the primary model's confidence.
5. If the prediction is sufficiently confident, the primary prediction is accepted.
6. Otherwise, DINOv2-SVM performs secondary classification.
7. The learned adaptive gate evaluates whether to select the specialist prediction.
8. If the acceptance conditions are not satisfied, the system recommends expert review.
9. Grad-CAM can generate a visual explanation of the primary classifier's prediction.

---

## 🛠️ Technologies Used

| Category             | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| Deep Learning        | PyTorch                     |
| Machine Learning     | Scikit-learn                |
| Primary Model        | ConvNeXt-Tiny               |
| Secondary Model      | DINOv2 + SVM                |
| Adaptive Decision    | Learned Classification Gate |
| Explainable AI       | Grad-CAM                    |
| Web Application      | Streamlit                   |
| Model Hosting        | Hugging Face Hub            |
| Cloud Deployment     | Streamlit Community Cloud   |
| Model Development    | Kaggle, Jupyter Notebook    |
| Version Control      | Git, GitHub                 |

---

## 📂 Project Structure

```text
OralSense-AI/
│
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   └── 02_Model_Training.ipynb
│
├── streamlit/
│   ├── app.py
│   ├── inference.py
│   ├── setup_models.py
│   ├── requirements.txt
│   │
│   └── assets/
│       └── styles.css
│
├── models/
│   └── README.md
│
├── assets/
│   ├── architecture.png
│   ├── confusion_matrix.png
│   └── gradcam_results.png
│
├── .gitignore
├── README.md
└── LICENSE
```

**Note:** Trained model weights are hosted externally on Hugging Face rather than committed directly to GitHub.

---

## 📊 Model Evaluation

The system was evaluated on an internal held-out test set containing **485 images**.

| Model / System           | Accuracy | Coverage |
| ------------------------ | -------: | -------: |
| ConvNeXt-Tiny            |   95.46% |     100% |
| ConvNeXt + Learned Gate  |   96.49% |     100% |
| ConvNeXt + Referral      |   99.49% |   81.65% |
| Complete Adaptive System |   99.01% |   83.51% |

### Evaluation Highlights

* **Primary classifier accuracy:** 95.46%
* **Learned-gate accuracy:** 96.49%
* **Complete adaptive system accepted accuracy:** 99.01%
* **Adaptive prediction coverage:** 83.51%
* **Images referred for expert review:** 80 out of 485

**Important:** Accuracy for systems incorporating referral is calculated only on accepted predictions. These results are from internal research evaluation and do not establish clinical diagnostic performance.

---

## 🌐 Cloud Deployment

OralSense AI uses **Streamlit Community Cloud** for web hosting and **Hugging Face Hub** for trained model storage.

### Deployment Workflow

```text
                GitHub Repository
                        |
                        v
              Streamlit Community Cloud
                        |
                        v
                 Streamlit Application
                        |
                        v
                 Model Setup Script
                        |
                        v
                  Hugging Face Hub
                        |
                        v
               Download Model Artifacts
                        |
                        v
                Initialize ML Pipeline
                        |
                        v
                 Image Classification
                        |
                        v
                Prediction / Referral
                        |
                        v
                Grad-CAM Explanation
```

### Deployment Configuration

| Configuration    | Value                        |
| ---------------- | ---------------------------- |
| Hosting Platform | Streamlit Community Cloud    |
| Python Version   | 3.11                         |
| Main Application | `streamlit/app.py`           |
| Dependencies     | `streamlit/requirements.txt` |
| Model Setup      | `streamlit/setup_models.py`  |
| Model Storage    | Hugging Face Hub             |

### Model Repository

The trained model artifacts are hosted at:

**[OralSense AI — Hugging Face Models](https://huggingface.co/Divyasri28/OralSense-AI-Models/tree/main)**

The model setup script retrieves the required files when preparing the application environment.

---

## 💻 Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Divyasri-M-2006/OralSense-AI.git
cd OralSense-AI
```

### 2. Install Dependencies

Python 3.11 is recommended.

```bash
python -m pip install -r streamlit/requirements.txt
```

### 3. Download Model Artifacts

The application uses `setup_models.py` to prepare the trained model files.

From the project root, run:

```bash
python streamlit/setup_models.py
```

The script retrieves the required model artifacts from Hugging Face.

### 4. Run the Application

```bash
cd streamlit
streamlit run app.py
```

The application will open in your browser.

---

## 🖥️ Web Application Features

The Streamlit application includes five main sections:

**Home:** Introduces the project, model architecture, and research objectives.

**Analyze Image:** Allows users to upload oral images, execute the classification pipeline, and view prediction results.

**Our Technology:** Explains the ConvNeXt-Tiny, DINOv2-SVM, and adaptive decision architecture.

**Performance:** Presents internal model evaluation results and comparisons.

**Project Info:** Describes the research objectives, dataset preparation, and limitations.

### Grad-CAM Explainability

Grad-CAM highlights image regions that influence ConvNeXt-Tiny's classification output.

The generated visualizations help users explore model behavior but do not represent clinically verified lesion locations.

---

## 🔬 Research Contributions

The project investigates:

1. Confidence-aware oral image classification.
2. Conditional specialist inference using DINOv2-SVM.
3. Learned adaptive switching between classification models.
4. Referral-based handling of uncertain predictions.
5. Grad-CAM-based interpretation of primary model predictions.
6. Deployment of a multi-model deep learning pipeline through a web application.

---

## 🚀 Future Enhancements

* External validation using independently collected oral image datasets.
* Improved calibration and uncertainty estimation.
* Additional model robustness and generalization studies.
* Performance optimization for cloud inference.
* Enhanced model explainability and user guidance.

---

## ⚠️ Disclaimer

OralSense AI is developed strictly for **academic research and educational purposes**.

It is not a medical device or clinically validated diagnostic system.

Predictions, confidence scores, and Grad-CAM visualizations must not be used to diagnose, confirm, or rule out oral diseases.

For medical concerns, consult a qualified healthcare professional.

---

## ⭐ Project Links

**🌐 Live Web Application:** [OralSense AI](https://oralsense-ai.streamlit.app)

**💻 GitHub Repository:** [OralSense-AI](https://github.com/Divyasri-M-2006/OralSense-AI)

**🤗 Trained Model Artifacts:** [Hugging Face](https://huggingface.co/Divyasri28/OralSense-AI-Models)

---

**Built with Python, PyTorch, Scikit-learn, Streamlit, and Hugging Face.**
