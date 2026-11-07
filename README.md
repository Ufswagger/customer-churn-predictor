# Customer Churn Predictor

Live app: https://your-app.streamlit.app

Predicts which telecom customers are likely to leave, and suggests actions to keep them.

## Key findings

- Month-to-month customers churn at about 43%, compared with about 3% on two-year contracts
- Most churn happens in the first year: about 48% for customers in months 1–12, falling to about 9% for months 49–72
- Electronic check users churn at about 45%, compared with about 15–19% for the other payment methods
- Fiber optic customers churn at about 42%, compared with about 19% on DSL
- Customers without tech support churn at about 42%, compared with about 15% for those who have it

Most fiber optic customers are on month-to-month contracts, so contract type and internet service overlap. The model helps separate these factors.

## Model

Logistic Regression (C = 1) with balanced class weights. On the held-out test set it catches 80% of customers who leave (recall 0.80) and flags 310 stayed customers as likely to churn (precision 0.49 for churners).

The model was chosen over Decision Tree and Random Forest after 5-fold cross-validation on the training data. Logistic Regression had the highest recall (0.805), and tuned Random Forest was close (0.797).

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `telco_churn.csv` in the same folder as `app.py`. The app trains the model on startup, so no saved model file is needed.

## Repository contents

- `app.py`: Streamlit app
- `requirements.txt`: Python packages
- `telco_churn.csv`: dataset
- `Churn_Project.ipynb`: notebook with EDA, modeling and evaluation
- `README.md`: this file

## Built with

Python, pandas, scikit-learn, Streamlit. Data from Kaggle (Telco Customer Churn).
