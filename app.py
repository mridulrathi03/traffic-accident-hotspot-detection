import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import plotly.express as px

# --- Configuration ---
st.set_page_config(page_title="Traffic Accident Hotspot Analysis", layout="wide")

# 1. Data Loading (Simulated or Real)
@st.cache_data
def load_data():
    """
    Simulates loading data if no file is uploaded.
    In a real scenario, you would replace this with pd.read_csv().
    """
    # Dummy data of India 
    data = {
        'latitude': np.random.uniform(18.5, 28.7, 100),  # Rough range for central/north India
        'longitude': np.random.uniform(72.8, 77.2, 100),
        'severity': np.random.choice(['Minor', 'Major', 'Fatal'], 100),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 100),
        'weather': np.random.choice(['Clear', 'Rainy', 'Foggy'], 100)
    }
    return pd.DataFrame(data)

# 2. Main App Interface
st.title("🚦 Traffic Accident Hotspot Analysis")
st.markdown("""
**Project by:** MD Aarish Qureshi, Mridul Rathi, Atharva Gaikwad, Kailash Sharma  
**Objective:** Identify high-risk geographical locations and visualize accident trends.
""")
st.write("---")

# Sidebar for Controls
st.sidebar.header("Filter Options")
uploaded_file = st.sidebar.file_uploader("Upload Accident Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Ensure columns exist (basic error handling)
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("Dataset must contain 'latitude' and 'longitude' columns.")
        st.stop()
else:
    st.sidebar.info("Using generated dummy data for demonstration.")
    df = load_data()

# Show Raw Data Preview
with st.expander("View Raw Data"):
    st.dataframe(df.head())

# --- 3. Data Visualization (EDA) ---
st.subheader("📊 Exploratory Data Analysis")
col1, col2 = st.columns(2)

with col1:
    # Trend by Severity
    fig_sev = px.bar(df, x='severity', title="Accident Counts by Severity", color='severity')
    st.plotly_chart(fig_sev, use_container_width=True)

with col2:
    # Trend by Time of Day
    if 'time_of_day' in df.columns:
        fig_time = px.pie(df, names='time_of_day', title="Accidents by Time of Day")
        st.plotly_chart(fig_time, use_container_width=True)

# --- 4. Hotspot Identification (Clustering Algorithm) ---
# As per PPT: Using Clustering (K-Means) to find hotspots [cite: 128]
st.subheader("📍 Hotspot Identification (K-Means Clustering)")

# User inputs for algorithm
num_clusters = st.sidebar.slider("Number of Hotspots (Clusters)", 2, 10, 5)

# Preparing coordinates
X = df[['latitude', 'longitude']].dropna()

if len(X) > num_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_

    st.success(f"Identified {num_clusters} accident hotspots using K-Means clustering.")
else:
    st.warning("Not enough data points for clustering.")

# --- 5. Interactive Map Visualization ---
# Using Folium as specified in tools [cite: 90]
st.subheader("🗺️ Geospatial Hotspot Map")

# Base map centered on average coordinates
m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=6)

# Plotting individual accidents
for idx, row in df.iterrows():
    # Color coding based on severity if available
    color = 'red' if row.get('severity') == 'Fatal' else 'blue'
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"Severity: {row.get('severity', 'N/A')}"
    ).add_to(m)

# Plotting Hotspot Centers (The "Clusters")
if 'cluster' in df.columns:
    for center in centers:
        folium.Marker(
            location=[center[0], center[1]],
            icon=folium.Icon(color='black', icon='info-sign', prefix='fa'),
            tooltip="High Risk Hotspot Center"
        ).add_to(m)

st_folium(m, width=800, height=500)

# --- 6. Recommendations ---
st.subheader("📢 Actionable Insights")
st.markdown("""
* **High Risk Zones:** The black markers on the map indicate mathematically calculated centers of accident clusters.
* **Targeted Action:** Traffic authorities should prioritize these zones for patrol deployment[cite: 102].
* **Infrastructure:** Engineers should inspect these specific lat/long coordinates for road defects[cite: 104].
""")