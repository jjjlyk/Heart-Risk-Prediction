import streamlit as st
import joblib
import pandas as pd
import os
import importlib.util

# Load models (if you want to use them directly elsewhere)
ad_optimization_q_table = joblib.load('models/ad_optimization_q_table.joblib')
association_rules_dataframe = joblib.load('models/association_rules_dataframe.joblib')
customer_segmentation_kmeans = joblib.load('models/customer_segmentation_kmeans.joblib')

st.title("Business Analytics Model Showcase")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "Customer Segmentation (Clustering)",
        "Market Basket Analysis (Association Rules)",
        "Reinforcement Learning (Conceptual Demo)"
    ]
)

models_dir = os.path.join(os.path.dirname(__file__), "models")
data_dir = os.path.join(os.path.dirname(__file__), "data")
sample_data_path = os.path.join(data_dir, "online_retail.csv")

if model_choice == "Customer Segmentation (Clustering)":
    st.header("Customer Segmentation using Clustering")
    model_path = os.path.join(models_dir, "clustering_model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "clustering_model.pkl")
    if os.path.exists(model_path):
        clustering_model = joblib.load(model_path)
        st.success("Clustering model loaded.")
        st.write("Upload customer data (CSV) for segmentation or use sample data:")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        use_sample = st.checkbox("Use sample data (online_retail.csv)")
        if uploaded_file or use_sample:
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(sample_data_path)
            st.write("Input Data:", df.head())
            if st.button("Predict Segments"):
                segments = clustering_model.predict(df)
                df['Segment'] = segments
                st.write("Segmented Data:", df)
    else:
        st.error("Clustering model not found.")

elif model_choice == "Market Basket Analysis (Association Rules)":
    st.header("Market Basket Analysis using Association Rules")
    model_path = os.path.join(models_dir, "association_rules_model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(models_dir, "association_rules_model.pkl")
    if os.path.exists(model_path):
        association_rules = joblib.load(model_path)
        st.success("Association rules loaded.")
        st.write("Sample Association Rules:")
        st.dataframe(association_rules)
        st.write("You can also explore the sample data (online_retail.csv):")
        if os.path.exists(sample_data_path):
            df = pd.read_csv(sample_data_path)
            st.write(df.head())
    else:
        st.error("Association rules model not found.")

elif model_choice == "Reinforcement Learning (Conceptual Demo)":
    st.header("Reinforcement Learning: Conceptual Demonstration")
    rl_demo_path = os.path.join(models_dir, "reinforcement_learning_demo.py")
    if os.path.exists(rl_demo_path):
        spec = importlib.util.spec_from_file_location("rl_demo", rl_demo_path)
        rl_demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rl_demo)
        if hasattr(rl_demo, "run_demo"):
            rl_demo.run_demo(st)
        else:
            st.warning("No 'run_demo(st)' function found in RL demo script.")
    else:
        st.error("Reinforcement learning demo script not found.")