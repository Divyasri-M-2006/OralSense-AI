# 🧠 OralSense AI — Trained Model Artifacts

This directory documents the trained machine learning and deep learning model artifacts used by the **OralSense AI** application.

OralSense AI combines ConvNeXt-Tiny, DINOv2-SVM, and a learned adaptive decision policy for confidence-aware oral image classification.

## 📦 Required Model Files

| Model File                         | Purpose                                        |
| ---------------------------------- | ---------------------------------------------- |
| `ConvNeXtTiny_best.pth`            | Trained ConvNeXt-Tiny primary image classifier |
| `DINOv2SVM_best.joblib`            | Trained SVM-based secondary classifier         |
| `OralSense_Adaptive_Policy.joblib` | Learned adaptive decision policy               |
| `dinov2_vits14_pretrain.pth`       | Pretrained DINOv2 ViT-S/14 weights             |

The DINOv2 source code is also required to initialize the feature extraction model.

## 🤗 Model Hosting

The trained model artifacts are hosted on **Hugging Face Hub**.

**Model Repository:** [OralSense-AI Models](https://huggingface.co/Divyasri28/OralSense-AI-Models/tree/main)

Model weights are stored externally to avoid committing large binary files directly to GitHub.

## ⚙️ Automatic Model Setup

The application uses the following script to prepare the required model files:

`streamlit/setup_models.py`

During setup, the script:

1. Checks for the required model artifacts.
2. Downloads missing model files from Hugging Face Hub.
3. Prepares the DINOv2 source code.
4. Makes the model artifacts available to the inference pipeline.

## 💻 Manual Setup

To prepare the models locally, navigate to the project root and run:

```bash
python streamlit/setup_models.py
```

The script downloads the required model artifacts.

Alternatively, the Streamlit application automatically invokes model setup during initialization.

## 🌐 Cloud Deployment

The OralSense AI website is hosted using **Streamlit Community Cloud**.

During application startup:

1. Streamlit loads the application source code from GitHub.
2. The model setup script retrieves the required files from Hugging Face Hub.
3. The trained models become available for inference.
4. Users can access the image classification interface.

**Live Application:** [OralSense AI](https://oralsense-ai.streamlit.app)

## 📁 Model Storage

Downloaded model files are stored in the application's model directory.

These files are excluded from Git version control.

Only model documentation, source code, and setup scripts are maintained in the GitHub repository.

## ⚠️ Important Notes

* The model files are intended for academic research and educational use.
* The trained weights are not committed directly to GitHub.
* Internet access is required when downloading model artifacts for the first time.
* Model predictions are not clinically validated diagnoses.
* The application should not be used as a substitute for professional dental assessment.
