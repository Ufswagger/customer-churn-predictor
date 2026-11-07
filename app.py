import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

# Best C from grid_lr.best_params_["clf__C"] in the notebook.
BEST_C = 1


@st.cache_resource
def load_model():
    """Load the CSV, clean it the same way as the notebook, and train the model.

    Training here (instead of loading a saved file) means the app only needs
    telco_churn.csv and requirements.txt to run.
    """
    df = pd.read_csv("telco_churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).drop(columns="customerID")

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns="Churn")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # Same model as the notebook's final choice: Logistic Regression with balanced weights.
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=BEST_C, class_weight="balanced", max_iter=1000)),
    ])
    model.fit(X, y)
    return model, X


model, X = load_model()

st.title("Customer Churn Predictor")
st.write("Enter a customer's details to see their churn risk and suggested actions.")

col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    contract = st.selectbox("Contract", sorted(X["Contract"].unique()))
with col2:
    payment = st.selectbox("Payment method", sorted(X["PaymentMethod"].unique()))
    internet = st.selectbox("Internet service", sorted(X["InternetService"].unique()))
    tech = st.selectbox("Tech support", sorted(X["TechSupport"].unique()))

# Start from the most common value for every column, then override the inputs.
row = X.mode().iloc[0].copy()
row["tenure"] = tenure
row["MonthlyCharges"] = monthly
row["TotalCharges"] = tenure * monthly   # estimate: the app has no billing history
row["Contract"] = contract
row["PaymentMethod"] = payment
row["InternetService"] = internet
row["TechSupport"] = tech

input_df = pd.DataFrame([row]).astype(X.dtypes.to_dict())[X.columns]

prob = model.predict_proba(input_df)[0, 1]

# Risk thresholds are a choice. Adjust them after checking the probabilities
# your test data produces.
if prob >= 0.6:
    risk = "High"
elif prob >= 0.3:
    risk = "Medium"
else:
    risk = "Low"

st.divider()
st.metric("Churn probability", f"{prob:.0%}")
st.subheader(f"Churn risk: {risk}")

actions = []
if contract == "Month-to-month":
    actions.append("Offer a discount for moving to a one-year contract.")
if payment == "Electronic check":
    actions.append("Offer a small discount for switching to automatic payment.")
if tech == "No":
    actions.append("Offer a free trial of tech support.")
if tenure < 12:
    actions.append("Schedule an onboarding check-in call in the first year.")
if not actions:
    actions.append("Keep regular check-ins. No urgent action needed.")

st.write("**Suggested actions to keep this customer:**")
for action in actions:
    st.write(f"- {action}")
