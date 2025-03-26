# Cardiovascular Disease Prediction System

![Cardiovascular Health](https://img.icons8.com/color/96/000000/heart-health.png)

A machine learning project to predict cardiovascular disease risk using patient health metrics, comparing multiple classification algorithms.

## Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [License](#license)

## Dataset
The dataset contains 70,000 patient records with 12 features including:
- Demographic information (age, gender, height, weight)
- Blood pressure measurements (systolic/diastolic)
- Cholesterol and glucose levels
- Lifestyle factors (smoking, alcohol intake, physical activity)
- Target variable: `cardio` (0=no disease, 1=disease)

Source: [Kaggle Cardiovascular Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

## Project Structure
cardio-disease-prediction/
├── data/
│ └── cardio_train.csv # Original dataset
├── notebooks/
│ └── Cardiovascular_Disease_Prediction.ipynb # Complete analysis notebook
├── images/ # Visualization exports
│ ├── target_dist.png
│ ├── age_dist.png
│ └── gender_dist.png
├── models/ # Serialized trained models
├── README.md
└── requirements.txt
## Usage
jupyter notebook notebooks/Cardiovascular_Disease_Prediction.ipynb
