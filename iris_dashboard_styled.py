import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Iris Species Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLS = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

SPECIES_STYLE = {
    "Setosa": {"color": "#22d3ee", "emoji": "🌿"},
    "Versicolor": {"color": "#a78bfa", "emoji": "💜"},
    "Virginica": {"color": "#f472b6", "emoji": "🌺"},
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

/* Background: deep gradient with soft glowing spots */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 10%, rgba(34,211,238,0.12), transparent 40%),
        radial-gradient(circle at 85% 30%, rgba(244,114,182,0.12), transparent 40%),
        linear-gradient(160deg, #070b1a 0%, #0f0a24 50%, #071a1f 100%);
    color: #e2e8f0;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1036 0%, #0b1424 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] h2 {
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Hero header */
.hero {
    padding: 28px 32px;
    border-radius: 24px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    margin-bottom: 24px;
}
.eyebrow {
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    color: #94a3b8;
}
.hero h1 {
    margin: 6px 0 4px 0;
    font-size: 44px;
    font-weight: 700;
    background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p { color: #cbd5e1; margin: 0 0 14px 0; }
.chips span {
    display: inline-block;
    margin: 4px 8px 4px 0;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    background: rgba(34,211,238,0.12);
    border: 1px solid rgba(34,211,238,0.35);
    color: #a5f3fc;
}

/* Glass cards */
.glass {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}
.glass h3 {
    margin-top: 0;
    font-size: 18px;
    color: #f1f5f9 !important;
}

/* Prediction card */
.result {
    text-align: center;
    padding: 26px 20px;
    border-radius: 20px;
    border: 1px solid var(--accent);
    background: radial-gradient(circle at top, color-mix(in srgb, var(--accent) 25%, transparent), rgba(255,255,255,0.03) 70%);
    box-shadow: 0 0 40px color-mix(in srgb, var(--accent) 35%, transparent);
}
.result .label { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 2px; }
.result .species { font-size: 46px; font-weight: 700; color: var(--accent); margin: 6px 0; }
.result .conf { color: #cbd5e1; font-size: 14px; }

/* Probability bars */
.bar-row { display: flex; align-items: center; gap: 12px; margin: 12px 0; }
.bar-row span { width: 100px; color: #e2e8f0; font-size: 14px; }
.bar-row b { width: 46px; text-align: right; color: #f1f5f9; font-size: 14px; }
.bar {
    flex: 1;
    height: 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
}
.fill { height: 100%; border-radius: 999px; }

/* Measurement tiles */
.tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.tile {
    padding: 14px;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
}
.tile small { color: #94a3b8; font-size: 12px; }
.tile div { font-size: 22px; font-weight: 600; color: #f8fafc; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def clean_name(label):
    return str(label).replace("Iris-", "").capitalize()


@st.cache_resource
def load_models_and_data():
    df = pd.read_csv("Iris.csv")
    df = df.drop(columns=["Id"], errors="ignore")
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    with open("iris_metrics.json") as f:
        metrics = json.load(f)
    return df, model, scaler, metrics


df, model, scaler, metrics = load_models_and_data()
df["Species"] = df["Species"].map(clean_name)
classes = [clean_name(c) for c in model.classes_]

# Hero header
st.markdown(
    f"""
<div class="hero">
<div class="eyebrow">CodeAlpha · Data Science Internship · Task 1</div>
<h1>Iris Species Studio</h1>
<p>Drag the sliders to measure a flower. The model predicts its species instantly.</p>
<div class="chips">
<span>🧠 {metrics["model"]}</span>
<span>🎯 Test accuracy {metrics["test_accuracy"]:.2f}%</span>
<span>🌸 150 flowers</span>
<span>🌱 3 species</span>
</div>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("🎛️ Measurements")
st.sidebar.caption("Set the four measurements of your flower (cm).")
sepal_length = st.sidebar.slider(
    "Sepal length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()),
    float(df["SepalLengthCm"].mean()), step=0.1,
)
sepal_width = st.sidebar.slider(
    "Sepal width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()),
    float(df["SepalWidthCm"].mean()), step=0.1,
)
petal_length = st.sidebar.slider(
    "Petal length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()),
    float(df["PetalLengthCm"].mean()), step=0.1,
)
petal_width = st.sidebar.slider(
    "Petal width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()),
    float(df["PetalWidthCm"].mean()), step=0.1,
)

# Prediction
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURE_COLS
)
scaled_input = scaler.transform(input_data)
probabilities = model.predict_proba(scaled_input)[0]
best_idx = probabilities.argmax()
prediction = classes[best_idx]
style = SPECIES_STYLE.get(prediction, {"color": "#22d3ee", "emoji": "🌸"})
confidence = probabilities[best_idx] * 100

bars = "".join(
    f'<div class="bar-row"><span>{SPECIES_STYLE.get(name, {"emoji": ""})["emoji"]} {name}</span>'
    f'<div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{SPECIES_STYLE.get(name, {"color": "#22d3ee"})["color"]}"></div></div>'
    f"<b>{p * 100:.0f}%</b></div>"
    for name, p in zip(classes, probabilities)
)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown(
        f"""
<div class="result" style="--accent:{style["color"]}">
<div class="label">Predicted species</div>
<div class="species">{style["emoji"]} {prediction}</div>
<div class="conf">Confidence: {confidence:.0f}%</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="glass">
<h3>Class probabilities</h3>
{bars}
</div>
""",
        unsafe_allow_html=True,
    )

    tiles = "".join(
        f'<div class="tile"><small>{label}</small><div>{value:.2f} cm</div></div>'
        for label, value in [
            ("Sepal length", sepal_length),
            ("Sepal width", sepal_width),
            ("Petal length", petal_length),
            ("Petal width", petal_width),
        ]
    )
    st.markdown(
        f"""
<div class="glass">
<h3>Your flower</h3>
<div class="tiles">{tiles}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown('<div class="glass"><h3>Where the species live</h3></div>', unsafe_allow_html=True)
    plot_df = df.rename(columns={"PetalLengthCm": "Petal length", "PetalWidthCm": "Petal width"})
    st.scatter_chart(
        plot_df, x="Petal length", y="Petal width", color="Species", height=340,
    )

    st.markdown('<div class="glass"><h3>Species statistics</h3></div>', unsafe_allow_html=True)
    species_selected = st.selectbox("Choose a species", classes, index=int(best_idx))
    filtered = df[df["Species"] == species_selected].drop(columns=["Species"])
    st.dataframe(
        filtered.describe().T[["mean", "min", "max", "std"]].round(2),
    )

st.markdown(
    '<p style="text-align:center;color:#64748b;margin-top:30px;">🚀 Built for the CodeAlpha Data Science Internship</p>',
    unsafe_allow_html=True,
)
