# Fake Job Posting Detector

A machine learning system that detects fraudulent job postings using NLP.

## What it does
Analyses job postings and predicts whether they are real or fraudulent
using a combination of TF-IDF text features and numeric signals.

## Key Results
| Model | ROC-AUC | Recall (Fake) | Precision (Fake) |
|-------|---------|---------------|------------------|
| Logistic Regression | 0.983 | 88% | 46% |
| Random Forest | 0.989 | 56% | 99% |

## Key Finding
Fake jobs average 300 words in description vs 800 for real jobs.
Part-time postings are 2x more likely to be fraudulent (9.3% vs 4.8% average).

## Tech Stack
Python · NLTK · TF-IDF · Scikit-learn · SHAP · Streamlit · Plotly

## Dataset
[Real or Fake Job Postings](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
— 17,880 job postings, 4.84% fraudulent

## Project Structure
```
notebooks/01_eda.ipynb        → Exploratory data analysis
notebooks/02_features.ipynb   → NLP feature engineering  
notebooks/03_model.ipynb      → Model training & evaluation
app.py                        → Streamlit dashboard
```

## How to run locally
```bash
git clone https://github.com/CHarshithReddy/fake-job-detector.git
cd fake-job-detector
pip install -r requirements.txt
streamlit run app.py
```
