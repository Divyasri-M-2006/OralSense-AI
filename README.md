# OralSense AI — Intelligent Oral Lesion Classification

An AI-powered oral lesion classification system integrating deep learning, machine learning, adaptive prediction, and Grad-CAM visualization through an interactive Streamlit web application.

## Project Overview

OralSense AI is a research-oriented machine learning project designed to classify oral lesion images and provide visual explanations of model predictions.

The project includes:

- Dataset preprocessing and preparation
- Deep learning model training and evaluation
- ConvNeXt and DINOv2-based classification components
- Adaptive prediction policy
- Streamlit-based user interface
- Grad-CAM visualization for model interpretability

## Technologies Used

Python | PyTorch | Scikit-learn | Streamlit | Jupyter Notebook | Kaggle

## Project Structure

- `notebooks/` — Dataset preprocessing and model training
- `streamlit/` — Web application and inference code
- `models/` — Model download and setup instructions
- `assets/` — Architecture diagrams and evaluation results

## Installation

Clone the repository:

```bash
git clone https://github.com/DIVYASRI-M-2006/OralSense-AI.git
cd OralSense-AI
```

Install dependencies:

```bash
pip install -r streamlit/requirements.txt
```

Model setup instructions will be added after model hosting is configured.

## Run the Application

After installing dependencies and setting up the trained models:

```bash
cd streamlit
streamlit run app.py
```

## Future Deployment

The application is planned for deployment using Render to make it accessible through a public website.

## Disclaimer

OralSense AI is intended for research and educational purposes. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional.