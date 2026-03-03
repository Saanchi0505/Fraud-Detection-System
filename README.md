# 🛡️ Fraud Detection System (End-to-End ML Deployment)

An end-to-end machine learning system for detecting fraudulent financial transactions using the BankSim dataset.  
The project covers data preprocessing, imbalanced classification handling, threshold optimization, and real-time deployment using Flask.

---

## 📌 Project Overview

This project builds a fraud detection model on a highly imbalanced dataset (≈1–2% fraud cases).  
The objective is to maximize fraud detection (recall) while maintaining acceptable precision through threshold tuning.

The model is deployed as a Flask web application where users can input transaction details and receive real-time fraud probability predictions.

---

## 🚀 Features

- Exploratory Data Analysis (EDA)
- Imbalanced data handling using `scale_pos_weight`
- XGBoost-based fraud classification
- Precision–Recall threshold optimization
- End-to-end sklearn `Pipeline` (preprocessing + model)
- Real-time prediction using Flask API
- Clean web interface for transaction analysis

---

## 🧠 Technical Highlights

### Machine Learning

- XGBoost Classifier
- Stratified train-test split
- ROC-AUC evaluation
- Precision–Recall tradeoff analysis
- Business-driven threshold selection

### Preprocessing

- StandardScaler for numerical features
- OneHotEncoder for categorical features
- ColumnTransformer + Pipeline for training–inference consistency

### Deployment

- Flask REST API
- JSON-based prediction endpoint
- Serialized sklearn Pipeline (`full_pipeline.pkl`)
- Probability-based fraud decision logic

---

## 📊 Model Performance

- High fraud recall (~94% at threshold 0.8)
- Balanced precision–recall tradeoff
- Strong ROC-AUC score
- Validated prediction consistency between notebook and deployment

---

## 🌐 System Workflow

User Input (Web UI)
↓
Flask API
↓
Sklearn Pipeline

Scaling

Encoding

XGBoost Model
↓
Fraud Probability Output
↓
Threshold-Based Decision

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Flask
- HTML/CSS

---

## 📁 Project Structure

├── notebooks/
│ ├── 01_EDA.ipynb
│ ├── 02_Preprocessing.ipynb
│ ├── 03_Model_Training.ipynb
│ ├── 04_Model_Evaluation.ipynb
│ ├── 05_Threshold_Tuning.ipynb
│ └── 06_Model_Export.ipynb
│
├── models/
│ ├── full_pipeline.pkl
│ ├── threshold.pkl
│
├── templates/
│ └── index.html
│
├── app.py
└── README.md

---

## 🎯 Key Learning Outcomes

- Handling highly imbalanced classification problems
- Understanding precision–recall tradeoffs
- Preventing feature mismatch using sklearn Pipeline
- Deploying ML models with Flask
- Building production-ready ML systems
