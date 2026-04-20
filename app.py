import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from sklearn.cluster import KMeans
import plotly.express as px

# page setup
st.set_page_config(page_title="Traffic Accident Hotspot Analysis", layout="wide")

# loading data (dummy if no file uploaded)
@st.cache_data
def load_data():
    # just generating random data for demo
    data = {
        'latitude': np.random.uniform(18.5, 28.7, 100),
        'longitude': np.random.uniform(72.8, 77.2, 100),
        'severity': np.random.choice(['Minor', 'Major', 'Fatal'], 100),
        'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 100),
        'weather': np.random.choice(['Clear', 'Rainy', 'Foggy'], 100)
    }
    return pd.DataFrame(data)

# title
st.title("🚦 Traffic Accident Hotspot Analysis")

st.markdown("""
Project by: MD Aarish Qureshi, Mridul Rathi, Atharva Gaikwad, Kailash Sharma  

Goal: Find accident-prone areas and visualize the data
""")

st.write("---")

# sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# data handling
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # basic check
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.error("File must have latitude and longitude columns")
        st.stop()
else:
    st.sidebar.info("Using sample data")
    df = load_data()

# preview
with st.expander("See Data"):
    st.dataframe(df.head())

# ---------------- EDA ----------------
st.subheader("📊 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(df, x='severity', title="Severity Count", color='severity')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    if 'time_of_day' in df.columns:
        fig2 = px.pie(df, names='time_of_day', title="Time Distribution")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Clustering ----------------
st.subheader("📍 Finding Hotspots")

k = st.sidebar.slider("Number of clusters", 2, 10, 5)

X = df[['latitude', 'longitude']].dropna()

if len(X) > k:
    model = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = model.fit_predict(X)
    centers = model.cluster_centers_

    st.success(f"{k} hotspots found")
else:
    st.warning("Not enough data")

# ---------------- Map ----------------
st.subheader("🗺️ Map View")

m = folium.Map(
    location=[df['latitude'].mean(), df['longitude'].mean()],
    zoom_start=6
)

# plotting points
for _, row in df.iterrows():
    color = 'red' if row.get('severity') == 'Fatal' else 'blue'

    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        popup=f"Severity: {row.get('severity', 'N/A')}"
    ).add_to(m)

# plotting cluster centers
if 'cluster' in df.columns:
    for c in centers:
        folium.Marker(
            location=[c[0], c[1]],
            tooltip="Hotspot"
        ).add_to(m)

st_folium(m, width=800, height=500)

# ---------------- Insights ----------------
st.subheader("📢 Insights")

st.markdown("""
- Black markers show accident hotspot centers  
- These areas need more traffic control  
- Roads in these areas should be checked properly  
""")
