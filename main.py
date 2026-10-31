import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Config
st.set_page_config(
    page_title="Iris Flower Classification Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Dark Neon UI
st.markdown("""
    <style>
    .main {
        background-color: #0B0F19;
        color: #F8FAFC;
    }
    .stSidebar {
        background-color: #151C2C;
    }
    h1, h2, h3 {
        color: #00F5D4 !important;
    }
    .metric-card {
        background-color: #151C2C;
        border: 1px solid #38BDF8;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models_and_data():
    df = pd.read_csv("Iris.csv")
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    return df, model, scaler

df, model, scaler = load_models_and_data()

# Header
st.title("🌸 Interactive Iris Species Classification Dashboard")
st.markdown("### CodeAlpha Data Science Internship — Task 1 | Built with Streamlit & Scikit-Learn")
st.markdown("---")

# Sidebar Controls
st.sidebar.header("🎛️ Flower Measurement Sliders")
sepal_length = st.sidebar.slider("Sepal Length (cm)", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()), float(df["SepalLengthCm"].mean()))
sepal_width = st.sidebar.slider("Sepal Width (cm)", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()), float(df["SepalWidthCm"].mean()))
petal_length = st.sidebar.slider("Petal Length (cm)", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()), float(df["PetalLengthCm"].mean()))
petal_width = st.sidebar.slider("Petal Width (cm)", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()), float(df["PetalWidthCm"].mean()))

# Main Layout
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("📊 Dataset Overview & Exploratory Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("📈 Species Feature Comparison")
    species_selected = st.selectbox("Select Species to Inspect:", df["Species"].unique())
    filtered_df = df[df["Species"] == species_selected]
    st.write(filtered_df.describe().T[["mean", "min", "max", "std"]].style.format("{:.2f}"))

with col2:
    st.subheader("🔮 Real-Time Interactive Prediction")
    
    # Input DataFrame
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_cols)
    
    # Scale and Predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    
    # Display Result Card
    st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #FF2A85 !important; margin: 0;">Predicted Species</h3>
            <h1 style="color: #00F5D4 !important; font-size: 40px; margin: 10px 0;">{prediction}</h1>
            <p style="color: #CBD5E1; margin: 0;">SVM Model Accuracy: 96.67%</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Input Summary")
    st.json({
        "Sepal Length": f"{sepal_length} cm",
        "Sepal Width": f"{sepal_width} cm",
        "Petal Length": f"{petal_length} cm",
        "Petal Width": f"{petal_width} cm"
    })

# Footer
st.markdown("---")
st.markdown("🚀 *Developed for CodeAlpha Data Science Internship.*")