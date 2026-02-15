# Heart Disease Prediction using ML Classification Models

## Problem Statement

Heart disease is one of the leading causes of death globally. Early prediction based on clinical data can help doctors intervene sooner and improve patient outcomes. In this project, I've implemented six different ML classification models on the Heart Disease dataset and compared their performance. The trained models are served through a Streamlit web app.

## Dataset Description

- **Name:** Heart Disease Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
- **Type:** Binary Classification
- **Features:** 13
- **Instances:** 1,025
- **Target:** `target` (0 = No Heart Disease, 1 = Heart Disease)

### Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age of the patient (years) | Numerical |
| sex | Sex (0 = Female, 1 = Male) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina (0/1) | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment (0-2) | Categorical |
| ca | Number of major vessels (0-4) | Numerical |
| thal | Thalassemia (0-3) | Categorical |

## Models and Evaluation Metrics

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | _run train_models.ipynb_ | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Random Forest (Ensemble) | | | | | | |
| XGBoost (Ensemble) | | | | | | |

> Run `model/train_models.ipynb` on Google Colab to get actual values. The notebook prints a ready-to-copy markdown table.

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.8098, AUC 0.9298. Good linear baseline - decent linear separability in the features. |
| Decision Tree | Accuracy 0.9854, MCC 0.9712. Captures non-linear patterns well. Depth limiting helps. |
| KNN | Accuracy 1.0000, F1 1.0000. Scaling helped a lot. Weighted distances better than uniform. |
| Naive Bayes | Accuracy 0.8293, Recall 0.8762. Independence assumption doesn't fully hold but still okay. |
| Random Forest (Ensemble) | Accuracy 1.0000, AUC 1.0000. Bagging reduces variance. Consistent. |
| XGBoost (Ensemble) | Accuracy 1.0000, MCC 1.0000. Strong precision-recall balance with boosting. |

## Project Structure

```
ML_Assignment2_A/
|-- app.py                  # Streamlit web application
|-- requirements.txt        # Python dependencies
|-- README.md               # This file
|-- model/
    |-- train_models.ipynb  # Model training notebook
    |-- *.pkl               # Saved model files (generated after training)
    |-- metrics.json        # Evaluation metrics (generated after training)
    |-- test_sample.csv     # Sample test data (generated after training)
```

## How to Run

### 1. Train the models (Google Colab)
1. Open `model/train_models.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `heart.csv` when prompted
3. Run all cells (~20-30 seconds)
4. Download the generated `.pkl`, `.json`, and `.csv` files
5. Place them in the `model/` folder

### 2. Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```