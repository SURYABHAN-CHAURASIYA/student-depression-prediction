# Student Depression Predictor

A Streamlit-based machine learning web app that predicts whether a student may be at risk of depression based on lifestyle, academic, and mental health related inputs.

## Features
- Predicts depression risk using a trained machine learning model
- Simple and interactive Streamlit user interface
- Risk score visualization using Plotly gauge chart
- Support suggestions for high-risk predictions
- Ready for GitHub and cloud deployment

## Project Files
- `app.py` - Main Streamlit application
- `train_model.py` - Model training script
- `models/depression_model.pkl` - Saved trained model
- `requirements.txt` - Python dependencies
- `student_depression_dataset.csv` - Dataset used for training

## How It Works
1. The model is trained using the dataset in `train_model.py`
2. The trained model is saved as `depression_model.pkl`
3. The Streamlit app loads the saved model and predicts the result based on user input

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/student-depression-predictor.git
cd student-depression-predictor
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Deployment
This app can be deployed on:
- Streamlit Community Cloud
- Render
- Hugging Face Spaces

## Disclaimer
This project is for educational purposes only and is not a medical diagnosis tool.