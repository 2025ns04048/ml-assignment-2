import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Assignment 2 - Classification Dashboard", layout="wide")

st.title("🩺 Machine Learning Classification & Deployment Dashboard")
st.markdown("**BITS M.Tech - Data Science & Engineering | Multi-Model Classification & Streamlit Deployment Dashboard")

# Sidebar for controls
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Model selection dropdown
model_name = st.sidebar.selectbox(
    "Select Classification Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Test Data Preview")
    st.dataframe(test_df.head())
    
    if "target" in test_df.columns:
        X_test = test_df.drop(columns=["target"])
        y_test = test_df["target"]
        
        # Load model and scaler
        model_filename = f"{model_name.lower().replace(' ', '_')}.pkl"
        try:
            model = joblib.load(model_filename)
            
            if model_name in ["Logistic Regression", "KNN", "Support Vector Machine"]:
                scaler = joblib.load("scaler.pkl")
                X_input = scaler.transform(X_test)
            else:
                X_input = X_test
                
            y_pred = model.predict(X_input)
            y_prob = model.predict_proba(X_input)[:, 1]
            
            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            st.markdown(f"### Evaluation Metrics for: **{model_name}**")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
                
            with col_b:
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose())
                
        except Exception as e:
            st.error(f"Error loading model or running prediction. Details: {e}")
    else:
        st.warning("Uploaded CSV must contain a 'target' column for evaluation.")
else:
    st.info("👈 Please upload your `test_data.csv` file in the sidebar to begin.")
