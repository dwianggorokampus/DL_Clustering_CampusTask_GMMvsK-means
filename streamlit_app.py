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

    /* Date Picker Fix - Aggressive */
    div[data-baseweb="popover"],
    div[data-baseweb="calendar"] {
        background-color: #161625 !important;
    }
    
    /* Force all inner divs to be dark to catch empty day slots and headers */
    div[data-baseweb="calendar"] div {
        background-color: #161625 !important;
        color: white !important;
    }
    
    /* Day Tiles (Buttons) - Transparent to let dark bg show through */
    div[data-baseweb="calendar"] button {
        background-color: transparent !important;
        color: white !important;
    }
    
    /* Hover State */
    div[data-baseweb="calendar"] button:hover {
        background-color: #2d2d44 !important;
        cursor: pointer;
    }
    
    /* Selected Day */
    div[data-baseweb="calendar"] button[aria-selected="true"] {
        background-color: #00d4ff !important;
        color: white !important;
    }
    
    /* Month/Year Dropdowns (if any) */
    div[data-baseweb="select"] div {
        background-color: #161625 !important;
        color: white !important;
    }
    
    /* SVG Icons (Arrows) */
    div[data-baseweb="calendar"] svg {
        fill: white !important;
        color: white !important;
    }
    
    /* Fix White Header Bar */
    header[data-testid="stHeader"] {
        background-color: #1e1e2f !important;
    }
    
    /* Hide the top decoration bar if needed */
    div[data-testid="stDecoration"] {
        background-image: none;
        background-color: #1e1e2f;
    }

    /* Fix Selectbox Dropdown Options (Popovers) */
    ul[data-baseweb="menu"] {
        background-color: #161625 !important;
    }
    li[data-baseweb="option"] {
        color: white !important;
    }
    /* Hover/Selected state in dropdown */
    li[data-baseweb="option"]:hover,
    li[data-baseweb="option"][aria-selected="true"] {
        background-color: #2d2d44 !important;
        color: #00d4ff !important;
    }
    /* Fix for the container of the dropdown list */
    div[data-baseweb="popover"] {
        background-color: #161625 !important;
    }
</style>
""", unsafe_allow_html=True)

# Load Models (Cached)
@st.cache_resource
def load_models():
    kmeans_model = None
    gmm_model = None
    scaler = None


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
    except Exception as e:
        st.error(f"Error loading models: {e}")
                
    return kmeans_model, gmm_model, scaler

kmeans_model, gmm_model, scaler = load_models()

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
page = st.sidebar.radio("Go to", ["Home", "Single Customer Grouping", "Batch Grouping", "Model Specs"])

if page == "Home":
    st.title("Customer Segmentation Dashboard")
    st.write("Welcome to the Customer Segmentation Tool. This dashboard provides an overview of your retail data.")
    
    if os.path.exists('Online Retail.xlsx'):
        try:
            @st.cache_data
            def load_data():
                df = pd.read_excel('Online Retail.xlsx')
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
                return df

            with st.spinner("Loading dataset..."):
                df = load_data()
            
            # KPIs
            total_sales = df['TotalPrice'].sum()
            total_orders = df['InvoiceNo'].nunique()
            total_customers = df['CustomerID'].nunique()
            avg_order_value = total_sales / total_orders
            
            st.subheader("Key Performance Indicators")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Total Sales", f"${total_sales:,.2f}")
            kpi2.metric("Total Orders", f"{total_orders:,}")
            kpi3.metric("Total Customers", f"{total_customers:,}")
            kpi4.metric("Avg Order Value", f"${avg_order_value:.2f}")
            
            st.divider()
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sales Trend Over Time")
                # Group by month
                sales_trend = df.set_index('InvoiceDate').resample('M')['TotalPrice'].sum().reset_index()
                st.line_chart(sales_trend, x='InvoiceDate', y='TotalPrice')
                
            with col2:
                st.subheader("Top 10 Selling Products")
                top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_products)
                
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Top 10 Countries by Sales")
                country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(country_sales)
                
            with col4:
                st.subheader("Hourly Sales Distribution")
                df['Hour'] = df['InvoiceDate'].dt.hour
                hourly_sales = df.groupby('Hour')['TotalPrice'].sum()
                st.bar_chart(hourly_sales)
                
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"Error loading data for EDA: {e}")
    else:
        st.info("Please ensure 'Online Retail.xlsx' is in the application directory to view the dashboard.")
        st.info("This application uses K-Means and GMM models to segment customers based on RFM analysis.")

elif page == "Single Customer Grouping":
    st.title("Single Customer Grouping")
    
    st.write("Enter transaction details below. You can add multiple rows for a single customer.")
    
    col1, col2 = st.columns(2)
    with col1:
        snapshot_date = st.date_input("Snapshot Date", value=datetime.now())
    
    # Initialize session state for data editor if not exists
    if 'input_df' not in st.session_state:
        st.session_state.input_df = pd.DataFrame(
            [{'InvoiceDate': datetime.now().date(), 'Quantity': 10, 'UnitPrice': 10.0}]
        )

    # Data Editor for multiple rows
    edited_df = st.data_editor(
        st.session_state.input_df,
        num_rows="dynamic",
        column_config={
            "InvoiceDate": st.column_config.DateColumn("Invoice Date", required=True),
            "Quantity": st.column_config.NumberColumn("Quantity", min_value=1, required=True),
            "UnitPrice": st.column_config.NumberColumn("Unit Price", min_value=0.0, format="$%.2f", required=True)
        },
        hide_index=True
    )
        
    model_type = st.selectbox("Select Model", ["K-Means", "GMM"])
    
    if st.button("Group Customer"):
        if edited_df.empty:
            st.error("Please enter at least one transaction.")
        else:
            # Process Data
            try:
                df_input = edited_df.copy()
                df_input['InvoiceDate'] = pd.to_datetime(df_input['InvoiceDate'])
                df_input['TotalPrice'] = df_input['Quantity'] * df_input['UnitPrice']
                
                # Aggregation Logic
                max_date = df_input['InvoiceDate'].max()
                recency = (pd.to_datetime(snapshot_date) - max_date).days
                
                if recency < 0:
                    st.error(f"Error: Snapshot Date ({snapshot_date}) cannot be earlier than the latest Invoice Date ({max_date.date()}).")
                else:
                    frequency = len(df_input) # Count of transactions (rows)
                    monetary = df_input['TotalPrice'].sum()
                    
                    # Apply Manual Weights (Matching Batch Logic)
                    # Weights: Recency / 2, Frequency * 3, Monetary * 1.5
                    recency_weighted = recency / 2
                    frequency_weighted = frequency * 3
                    monetary_weighted = monetary * 1.5
                    
                    input_data = [[recency_weighted, frequency_weighted, monetary_weighted]]
                    
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
                        st.success(f"Group: **{prediction['name']}** (Cluster {prediction['cluster']})")
                        st.write(f"Model used: {prediction['model']}")
                        st.write(f"**Calculated Metrics (Aggregated):**")
                        st.write(f"- Recency: {recency} days (Snapshot - {max_date.date()})")
                        st.write(f"- Frequency: {frequency} transactions")
                        st.write(f"- Monetary: ${monetary:,.2f}")
                        st.caption(f"Weighted Input: R={recency_weighted:.2f}, F={frequency_weighted:.2f}, M={monetary_weighted:.2f}")
            except Exception as e:
                st.error(f"Error processing input data: {e}")

elif page == "Batch Grouping":
    st.title("Batch Grouping")
    
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
            
            if st.button("Run Grouping"):
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        if kmeans_model:
            st.header("K-Means Model")
            st.write(f"**Clusters:** {kmeans_model.n_clusters}")
            st.write(f"**Inertia:** {kmeans_model.inertia_:.2f}")
            
            # Try to load original data for metrics
            if os.path.exists('Online Retail.xlsx'):
                try:
                    # Load and process data (Cached for performance)
                    @st.cache_data
                    def get_training_metrics():
                        df = pd.read_excel('Online Retail.xlsx')
                        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                        
                        # Use max date + 1 day as snapshot to match training context (avoiding 5000+ day recency)
                        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
                        
                        rfm = calculate_rfm(df, snapshot_date)
                        
                        # Preprocessing
                        rfm_weighted = rfm[["Recency", "Frequency", "Monetary"]].copy()
                        rfm_weighted["Recency"] = rfm_weighted["Recency"] / 2
                        rfm_weighted["Frequency"] = rfm_weighted["Frequency"] * 3
                        rfm_weighted["Monetary"] = rfm_weighted["Monetary"] * 1.5
                        
                        if scaler:
                            rfm_scaled = scaler.transform(rfm_weighted)
                        else:
                            rfm_scaled = rfm_weighted.values
                            
                        return rfm, rfm_scaled

                    rfm_real, rfm_scaled_real = get_training_metrics()
                    
                    labels = kmeans_model.predict(rfm_scaled_real)
                    
                    # Silhouette Score
                    # Sampling for performance if > 10k samples
                    if len(rfm_scaled_real) > 10000:
                        score = silhouette_score(rfm_scaled_real, labels, sample_size=10000)
                        st.metric("Silhouette Score (Real Data - Sampled)", f"{score:.4f}")
                    else:
                        score = silhouette_score(rfm_scaled_real, labels)
                        st.metric("Silhouette Score (Real Data)", f"{score:.4f}")
                        
                    # Visualization using REAL data
                    st.subheader("Cluster Visualization (Real Data)")
                    rfm_real['Cluster'] = labels
                    rfm_real['Cluster_Name'] = [get_cluster_name(c) for c in labels]
                    
                    pairs = [
                        ('Recency', 'Frequency', 'Recency vs Frequency'),
                        ('Frequency', 'Monetary', 'Frequency vs Monetary'),
                        ('Recency', 'Monetary', 'Recency vs Monetary')
                    ]
                    
                    for x_col, y_col, title in pairs:
                        st.write(f"**{title}**")
                        st.scatter_chart(rfm_real, x=x_col, y=y_col, color='Cluster_Name')
                        
                except Exception as e:
                    st.warning(f"Could not calculate metrics from Online Retail.xlsx: {e}")
                    # Fallback to centroids
                    st.subheader("Cluster Centers (Centroids)")
                    centers_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=['Recency', 'Frequency', 'Monetary'])
                    centers_df['Cluster'] = [get_cluster_name(i) for i in range(len(centers_df))]
                    st.dataframe(centers_df.style.highlight_max(axis=0))
            else:
                # Silhouette Score (Training data not available)
                st.info("Silhouette Score requires 'Online Retail.xlsx' to be present.")

                st.subheader("Cluster Centers (Centroids)")
                centers_df = pd.DataFrame(kmeans_model.cluster_centers_, columns=['Recency', 'Frequency', 'Monetary'])
                centers_df['Cluster'] = [get_cluster_name(i) for i in range(len(centers_df))]
                st.dataframe(centers_df.style.highlight_max(axis=0))
                
                st.subheader("2D Visualizations (Centroids)")
                st.write("Since K-Means only stores centroids, here are the center points of each cluster:")
                
                # Pairwise Plots for K-Means
                pairs = [
                    ('Recency', 'Frequency', 'Recency vs Frequency'),
                    ('Frequency', 'Monetary', 'Frequency vs Monetary'),
                    ('Recency', 'Monetary', 'Recency vs Monetary')
                ]
                
                for x_col, y_col, title in pairs:
                    st.write(f"**{title}**")
                    st.scatter_chart(centers_df, x=x_col, y=y_col, color='Cluster', size=100)
                    st.caption(f"Shows the center point of each cluster for {x_col} and {y_col}.")
            
    with col2:
        if gmm_model:
            st.header("Gaussian Mixture Model")
            st.write(f"**Components:** {gmm_model.n_components}")
            st.write(f"**Covariance Type:** {gmm_model.covariance_type}")
            
            if hasattr(gmm_model, 'lower_bound_'):
                st.write(f"**Log-Likelihood:** {gmm_model.lower_bound_:.2f}")

            # Generate Synthetic Data
            try:
                n_samples = 200
                X_sampled, y_sampled = gmm_model.sample(n_samples)
                
                if len(set(y_sampled)) > 1:
                    syn_score = silhouette_score(X_sampled, y_sampled)
                    st.metric("Silhouette Score (Estimated)", f"{syn_score:.4f}")
                
                st.subheader("Cluster Visualization (Synthetic)")
                st.write(f"Generated {n_samples} points to visualize cluster distribution.")
                
                syn_df = pd.DataFrame(X_sampled, columns=['Recency', 'Frequency', 'Monetary'])
                syn_df['Cluster'] = [get_cluster_name(c) for c in y_sampled]
                
                # Pairwise Plots for GMM
                pairs = [
                    ('Recency', 'Frequency', 'Recency vs Frequency'),
                    ('Frequency', 'Monetary', 'Frequency vs Monetary'),
                    ('Recency', 'Monetary', 'Recency vs Monetary')
                ]
                
                for x_col, y_col, title in pairs:
                    st.write(f"**{title}**")
                    st.scatter_chart(syn_df, x=x_col, y=y_col, color='Cluster')
                    st.caption(f"Synthetic distribution of clusters for {x_col} and {y_col}.")
                
            except Exception as e:
                st.warning(f"Could not generate synthetic visualization: {e}")

            st.subheader("Cluster Weights")
            weights_df = pd.DataFrame(gmm_model.weights_, columns=['Weight'])
            weights_df.index.name = 'Cluster'
            st.bar_chart(weights_df)
