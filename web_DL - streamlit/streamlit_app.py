import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
from sklearn.metrics import silhouette_score

# Set page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Custom CSS
st.markdown("""
<style>
    /* Global Text Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161625;
        border-right: 1px solid #333;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Text Elements */
    p, label, span, div {
        color: #e0e0e0 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00d4ff 0%, #005bea 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 212, 255, 0.4);
        color: white !important;
    }
    
    /* Inputs (Text, Number, Date) */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stDateInput>div>div>input {
        background-color: #2d2d44 !important;
        color: white !important;
        border: 1px solid #555 !important;
        border-radius: 6px;
    }
    .stTextInput>div>div>input:focus, 
    .stNumberInput>div>div>input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 1px #00d4ff !important;
    }
    
    /* Selectbox & Dropdowns */
    .stSelectbox>div>div>div {
        background-color: #2d2d44 !important;
        color: white !important;
        border: 1px solid #555 !important;
    }
    
    /* Radio Buttons */
    .stRadio>div {
        background-color: transparent;
    }
    .stRadio label {
        color: #ffffff !important;
        font-weight: 500;
    }
    
    /* Dataframes/Tables */
    [data-testid="stDataFrame"] {
        background-color: #232336;
        border: 1px solid #444;
        border-radius: 8px;
    }
    [data-testid="stDataFrame"] th {
        background-color: #161625 !important;
        color: #00d4ff !important;
    }
    [data-testid="stDataFrame"] td {
        color: #e0e0e0 !important;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #aaaaaa !important;
    }
    [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2rem !important;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] section {
        background-color: #232336;
        border-radius: 10px;
        padding: 20px;
        border: 1px dashed #555;
    }
    [data-testid="stFileUploader"] small {
        color: #aaaaaa !important;
    }
    
    /* Alerts (Success, Info, Warning, Error) */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #232336 !important;
        border: 1px solid #444;
        border-radius: 8px;
    }
    .stAlert p {
        color: white !important;
    }
    
    /* JSON Viewer */
    .stJson {
        background-color: #161625;
        padding: 10px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Load Models (Cached)
@st.cache_resource
def load_models():
    kmeans_model = None
    gmm_model = None
    scaler = None
    metrics = {}

    try:
        if os.path.exists('kmeans_model.pkl'):
            with open('kmeans_model.pkl', 'rb') as f:
                kmeans_model = pickle.load(f)
        
        if os.path.exists('gmm_model.pkl'):
            with open('gmm_model.pkl', 'rb') as f:
                gmm_model = pickle.load(f)
                
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
        if os.path.exists('model_metrics.json'):
            with open('model_metrics.json', 'r') as f:
                metrics = json.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        
    return kmeans_model, gmm_model, scaler, metrics

kmeans_model, gmm_model, scaler, metrics = load_models()

# Constants
CLUSTER_NAMES = {
    0: "Active Customers",
    1: "Inactive/Lost Customers",
    -1: "Noise"
}

def get_cluster_name(cluster_id):
    return CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")

def calculate_rfm(df, snapshot_date=None):
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    if snapshot_date is None:
        snapshot_date = datetime.now()
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)
    
    return rfm

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Single Prediction", "Batch Prediction", "Model Specs"])

if page == "Home":
    st.title("Customer Segmentation App")
    st.write("Welcome to the Customer Segmentation Tool. Use the sidebar to navigate.")
    st.info("This application uses K-Means and GMM models to segment customers based on RFM analysis.")

elif page == "Single Prediction":
    st.title("Single Customer Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days)", min_value=0.0, value=10.0)
    with col2:
        frequency = st.number_input("Frequency (count)", min_value=0.0, value=5.0)
    with col3:
        monetary = st.number_input("Monetary (value)", min_value=0.0, value=100.0)
        
    model_type = st.selectbox("Select Model", ["K-Means", "GMM"])
    
    if st.button("Predict"):
        input_data = [[recency, frequency, monetary]]
        
        # Note: Following app.py logic which does NOT explicitly weight single predictions before scaling
        if scaler:
            input_data_scaled = scaler.transform(input_data)
        else:
            input_data_scaled = input_data
            
        prediction = None
        if model_type == "K-Means" and kmeans_model:
            cluster = kmeans_model.predict(input_data_scaled)[0]
            name = get_cluster_name(cluster)
            prediction = {"cluster": cluster, "name": name, "model": "K-Means"}
        elif model_type == "GMM" and gmm_model:
            cluster = gmm_model.predict(input_data_scaled)[0]
            name = get_cluster_name(cluster)
            prediction = {"cluster": cluster, "name": name, "model": "GMM"}
        else:
            st.error("Selected model not loaded.")
            
        if prediction:
            st.success(f"Prediction: **{prediction['name']}** (Cluster {prediction['cluster']})")
            st.write(f"Model used: {prediction['model']}")
            st.write(f"Input: R={recency}, F={frequency}, M={monetary}")

elif page == "Batch Prediction":
    st.title("Batch Prediction")
    
    # Options Section
    st.subheader("Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        snapshot_option = st.radio("Snapshot Date", ["Today", "Custom"])
        if snapshot_option == "Custom":
            snapshot_date = st.date_input("Select Date")
            snapshot_date = pd.to_datetime(snapshot_date)
        else:
            snapshot_date = datetime.now()
            
    with col2:
        model_type = st.selectbox("Select Model", ["K-Means", "GMM"], key="batch_model")

    st.subheader("Upload Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)
                
            st.write("Preview of uploaded data:", raw_df.head())
            
            if st.button("Run Segmentation"):
                # RFM Calculation
                # Check if raw data has transaction columns or is already RFM
                if 'InvoiceDate' in raw_df.columns and 'Quantity' in raw_df.columns:
                    rfm_df = calculate_rfm(raw_df, snapshot_date=snapshot_date)
                    rfm_df.reset_index(inplace=True)
                    
                    # Weighted Scaling (Matches app.py batch logic)
                    rfm_weighted = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
                    rfm_weighted["Recency"] = rfm_weighted["Recency"] / 2
                    rfm_weighted["Frequency"] = rfm_weighted["Frequency"] * 3
                    rfm_weighted["Monetary"] = rfm_weighted["Monetary"] * 1.5
                    
                    if scaler:
                        rfm_scaled = scaler.transform(rfm_weighted)
                    else:
                        rfm_scaled = rfm_weighted.values
                else:
                    # Assume input is already RFM-like if columns missing? app.py logic:
                    rfm_df = raw_df.copy()
                    # Ensure columns exist
                    if all(col in rfm_df.columns for col in ["Recency", "Frequency", "Monetary"]):
                        rfm_weighted = rfm_df[["Recency", "Frequency", "Monetary"]].copy()
                        rfm_weighted["Recency"] = rfm_weighted["Recency"] / 2
                        rfm_weighted["Frequency"] = rfm_weighted["Frequency"] * 3
                        rfm_weighted["Monetary"] = rfm_weighted["Monetary"] * 1.5
                        
                        if scaler:
                            rfm_scaled = scaler.transform(rfm_weighted)
                        else:
                            rfm_scaled = rfm_weighted.values
                    else:
                        st.error("Uploaded file must contain 'InvoiceDate' and 'Quantity' OR 'Recency', 'Frequency', 'Monetary'.")
                        rfm_scaled = None

                # Prediction
                if rfm_scaled is not None:
                    selected_model = kmeans_model if model_type == "K-Means" else gmm_model
                    
                    if selected_model:
                        clusters = selected_model.predict(rfm_scaled)
                        rfm_df['Cluster'] = clusters
                        rfm_df['Cluster_Name'] = rfm_df['Cluster'].apply(get_cluster_name)
                        
                        # Silhouette Score
                        if len(set(clusters)) > 1 and len(rfm_scaled) > 2:
                            score = silhouette_score(rfm_scaled, clusters)
                            st.metric("Silhouette Score", f"{score:.4f}")
                        else:
                            st.warning("Not enough clusters/samples for Silhouette Score")
                            
                        st.write("Segmentation Results:", rfm_df.head(20))
                        
                        # Download
                        csv = rfm_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"segmented_{uploaded_file.name}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("Selected model not available.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Model Specs":
    st.title("Model Specifications")
    
    if kmeans_model:
        st.subheader("K-Means Model")
        st.json({
            "Algorithm": "K-Means",
            "n_clusters": getattr(kmeans_model, 'n_clusters', 'N/A'),
            "inertia": getattr(kmeans_model, 'inertia_', 'N/A'),
            "n_features": getattr(kmeans_model, 'n_features_in_', 'N/A'),
            "Silhouette Score": f"{metrics.get('kmeans_silhouette', 'N/A'):.4f}" if 'kmeans_silhouette' in metrics else "N/A"
        })
        
    if gmm_model:
        st.subheader("Gaussian Mixture Model")
        st.json({
            "Algorithm": "Gaussian Mixture Model",
            "n_components": getattr(gmm_model, 'n_components', 'N/A'),
            "covariance_type": getattr(gmm_model, 'covariance_type', 'N/A'),
            "n_features": getattr(gmm_model, 'n_features_in_', 'N/A')
        })
