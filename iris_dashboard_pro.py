"""Iris Studio: an analytics-style Streamlit dashboard for the Iris classification project."""

import json
import math
import time

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Iris Studio",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
FEATURES = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
FEATURE_LABELS = {
    "SepalLengthCm": "Sepal length (cm)",
    "SepalWidthCm": "Sepal width (cm)",
    "PetalLengthCm": "Petal length (cm)",
    "PetalWidthCm": "Petal width (cm)",
}
SPECIES_COLORS = {"Setosa": "#6366F1", "Versicolor": "#10B981", "Virginica": "#38BDF8"}
SPECIES_SCALE = alt.Scale(domain=list(SPECIES_COLORS), range=list(SPECIES_COLORS.values()))
BASELINE_ACCURACY = 100 / 3  # accuracy of guessing one of 3 species at random
PAGE_SIZES = [10, 25, 50]

# ----------------------------------------------------------------------------
# Styling: dark slate background, glass cards, indigo/emerald accents
# ----------------------------------------------------------------------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

[data-testid="stAppViewContainer"] { background: #0B0F19; color: #E2E8F0; }
[data-testid="stAppViewContainer"]::before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background:
        radial-gradient(circle at 10% 0%, rgba(99,102,241,0.18), transparent 35%),
        radial-gradient(circle at 90% 10%, rgba(16,185,129,0.10), transparent 30%);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { color: #F8FAFC !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0F172A; border-right: 1px solid rgba(148,163,184,0.12); }
.brand { font-size: 20px; font-weight: 700; color: #F8FAFC; margin: 4px 0 18px 0; }
.brand span { color: #6366F1; }

/* Bordered containers act as glass cards with a glow on hover */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(30,41,59,0.6) !important;
    border: 1px solid rgba(148,163,184,0.15) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    backdrop-filter: blur(10px);
    transition: border-color .2s ease, box-shadow .2s ease;
    margin-bottom: 12px;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 12px 40px rgba(99,102,241,0.18);
}

/* Plain HTML cards: lift and glow on hover */
.card {
    background: rgba(30,41,59,0.75);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
    margin-bottom: 12px;
}
.card:hover {
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.6);
    box-shadow: 0 12px 40px rgba(99,102,241,0.18);
}
.card h4 { margin: 0 0 12px 0; font-size: 15px; font-weight: 600; }

/* KPI row */
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 8px; }
.label { font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
.value { font-size: 30px; font-weight: 700; color: #F8FAFC; margin: 6px 0 4px 0; }
.delta { font-size: 13px; font-weight: 500; }
.up { color: #10B981; }
.down { color: #F43F5E; }
.flat { color: #94A3B8; }

/* Prediction card */
.result { text-align: center; border-color: var(--accent) !important; }
.result .species { font-size: 44px; font-weight: 700; color: var(--accent); margin: 6px 0; }
.conf { color: #CBD5E1; font-size: 14px; }

/* Probability bars */
.bar-row { display: flex; align-items: center; gap: 12px; margin: 12px 0; }
.bar-row span { width: 110px; font-size: 14px; color: #E2E8F0; }
.bar-row b { width: 46px; text-align: right; font-size: 14px; color: #F8FAFC; }
.bar { flex: 1; height: 10px; border-radius: 999px; background: rgba(148,163,184,0.15); overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }

/* Log table */
table.data { width: 100%; border-collapse: collapse; font-size: 14px; }
table.data th {
    text-align: left; color: #94A3B8; font-weight: 500;
    padding: 10px; border-bottom: 1px solid rgba(148,163,184,0.2);
}
table.data td { padding: 11px 10px; border-bottom: 1px solid rgba(148,163,184,0.08); color: #E2E8F0; }
table.data tr:hover td { background: rgba(99,102,241,0.08); }
.badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
.badge.ok { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.35); }
.badge.warn { background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.35); }

/* Responsive grid */
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 560px) {
    .kpis { grid-template-columns: 1fr; }
    .value { font-size: 24px; }
    .result .species { font-size: 34px; }
}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data and model loading
# ----------------------------------------------------------------------------
def clean_species(label):
    """Turn 'Iris-setosa' into 'Setosa'."""
    return str(label).replace("Iris-", "").capitalize()


@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    df["Species"] = df["Species"].map(clean_species)
    # The CSV has no dates, so spread the 150 flowers over the last 150 days (demo only).
    df["Date"] = pd.date_range(end=pd.Timestamp("2026-10-31"), periods=len(df), freq="D")
    return df


@st.cache_resource
def load_model():
    model = joblib.load("iris_model.pkl")
    scaler = joblib.load("iris_scaler.pkl")
    with open("iris_metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, metrics


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def style_chart(chart):
    """Make Altair charts readable on the dark background."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#94A3B8", titleColor="#94A3B8",
            gridColor="#1E293B", domainColor="#334155",
        )
        .configure_legend(labelColor="#CBD5E1", titleColor="#CBD5E1")
    )


def kpi_card(label, value, delta, cls):
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="delta {cls}">{delta}</div></div>'
    )


def row_html(r):
    badge = "ok" if r.Status == "Match" else "warn"
    return (
        f"<tr><td>#{int(r.Id)}</td><td>{r.Date:%d %b %Y}</td><td>{r.Species}</td>"
        f"<td>{r.PetalLengthCm:.1f}</td><td>{r.PetalWidthCm:.1f}</td>"
        f"<td>{r.Predicted}</td>"
        f'<td><span class="badge {badge}">{r.Status}</span></td></tr>'
    )


# ----------------------------------------------------------------------------
# Overview page
# ----------------------------------------------------------------------------
def render_overview(df, f, metrics, search):
    if f.empty:
        st.warning("No flowers match the current filters. Pick more species or a longer date range.")
        return

    # KPI cards: every trend number is computed from the data, not made up
    n = len(f)
    match_rate = (f["Status"] == "Match").mean() * 100
    review_count = int((f["Status"] == "Review").sum())
    avg_petal = f["PetalLengthCm"].mean()
    petal_gap = avg_petal - df["PetalLengthCm"].mean()
    petal_arrow, petal_cls = ("▲", "up") if petal_gap >= 0 else ("▼", "down")
    acc = metrics["test_accuracy"]

    st.markdown(
        '<div class="kpis">'
        + kpi_card("Flowers in view", f"{n}", f"{n / len(df) * 100:.0f}% of the dataset", "flat")
        + kpi_card(
            "Model test accuracy", f"{acc:.1f}%",
            f"▲ {acc - BASELINE_ACCURACY:.1f} pts above random guess", "up",
        )
        + kpi_card(
            "Avg petal length", f"{avg_petal:.2f} cm",
            f"{petal_arrow} {abs(petal_gap):.2f} cm vs all flowers", petal_cls,
        )
        + kpi_card("Match rate", f"{match_rate:.1f}%", f"{review_count} flagged for review", "flat")
        + "</div>",
        unsafe_allow_html=True,
    )

    # Charts
    left, right = st.columns([2, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### Petal length trend")
            trend = (
                f.set_index("Date")
                .groupby("Species")["PetalLengthCm"]
                .resample("W")
                .mean()
                .dropna()
                .reset_index()
            )
            line = (
                alt.Chart(trend)
                .mark_line(point=True, strokeWidth=2.5)
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("PetalLengthCm:Q", title="Avg petal length (cm)"),
                    color=alt.Color("Species:N", scale=SPECIES_SCALE),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Week of", format="%d %b %Y"),
                        "Species:N",
                        alt.Tooltip("PetalLengthCm:Q", title="Avg (cm)", format=".2f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(style_chart(line))

    with right:
        with st.container(border=True):
            st.markdown("#### Species mix")
            counts = f["Species"].value_counts().rename_axis("Species").reset_index(name="Count")
            donut = (
                alt.Chart(counts)
                .mark_arc(innerRadius=70, outerRadius=110, cornerRadius=6)
                .encode(
                    theta=alt.Theta("Count:Q"),
                    color=alt.Color("Species:N", scale=SPECIES_SCALE, legend=alt.Legend(orient="bottom")),
                    tooltip=["Species:N", "Count:Q"],
                )
                .properties(height=300)
            )
            st.altair_chart(style_chart(donut))

    # Log table with search and pagination
    st.markdown("#### Flower log")
    table = f.copy()
    query = search.strip()
    if query:
        mask = table["Species"].str.contains(query, case=False) | table["Id"].astype(str).str.contains(query)
        table = table[mask]

    with st.container(border=True):
        if table.empty:
            st.info("Nothing matches your search.")
        else:
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                page_size = st.selectbox("Rows per page", PAGE_SIZES)
            pages = math.ceil(len(table) / page_size)
            with c2:
                page_no = st.selectbox("Page", list(range(1, pages + 1)))

            start = (page_no - 1) * page_size
            chunk = table.sort_values("Date", ascending=False).iloc[start:start + page_size]
            body = "".join(row_html(r) for r in chunk.itertuples())
            st.markdown(
                '<table class="data"><thead><tr>'
                "<th>Flower</th><th>Date</th><th>Species</th><th>Petal L</th>"
                "<th>Petal W</th><th>Predicted</th><th>Status</th>"
                "</tr></thead><tbody>" + body + "</tbody></table>",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Showing {start + 1}–{min(start + page_size, len(table))} of {len(table)} flowers. "
                "Match means the model agrees with the recorded species. "
                "The model was trained on these flowers, so this is not a fair test."
            )

    # Export with a loading state
    with st.container(border=True):
        st.markdown("#### Export report")
        if st.button("Generate report", type="primary"):
            with st.spinner("Building your report…"):
                time.sleep(1.2)  # simulated work
            st.session_state["report_csv"] = f.to_csv(index=False)
            st.toast("Report ready")
        if "report_csv" in st.session_state:
            st.download_button(
                "⬇ Download CSV",
                st.session_state["report_csv"],
                file_name="iris_report.csv",
                mime="text/csv",
            )
        st.caption("CSV only for now. PDF export is not built yet.")


# ----------------------------------------------------------------------------
# Predict page
# ----------------------------------------------------------------------------
def render_predict(df, model, scaler, classes):
    st.markdown("#### Measure a flower")
    with st.container(border=True):
        cols = st.columns(2)
        values = {}
        for i, feat in enumerate(FEATURES):
            with cols[i % 2]:
                values[feat] = st.slider(
                    FEATURE_LABELS[feat],
                    float(df[feat].min()),
                    float(df[feat].max()),
                    float(df[feat].mean()),
                    step=0.1,
                )

    input_df = pd.DataFrame([values])[FEATURES]
    probs = model.predict_proba(scaler.transform(input_df))[0]
    best = int(probs.argmax())
    name = classes[best]
    color = SPECIES_COLORS.get(name, "#6366F1")

    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<div class="card result" style="--accent:{color}">'
            f'<div class="label">Predicted species</div>'
            f'<div class="species">{name}</div>'
            f'<div class="conf">Confidence {probs[best] * 100:.0f}%</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    with right:
        bars = "".join(
            f'<div class="bar-row"><span>{c}</span>'
            f'<div class="bar"><div class="fill" style="width:{p * 100:.1f}%;background:{SPECIES_COLORS[c]}"></div></div>'
            f"<b>{p * 100:.0f}%</b></div>"
            for c, p in zip(classes, probs)
        )
        st.markdown(f'<div class="card"><h4>Class probabilities</h4>{bars}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# App layout
# ----------------------------------------------------------------------------
df = load_data()
model, scaler, metrics = load_model()
classes = [clean_species(c) for c in model.classes_]

# Predictions for every flower, used by the log table and status badges
df["Predicted"] = [clean_species(p) for p in model.predict(scaler.transform(df[FEATURES]))]
df["Status"] = (df["Predicted"] == df["Species"]).map({True: "Match", False: "Review"})

# Sidebar navigation and filters
with st.sidebar:
    st.markdown('<div class="brand">🌸 Iris <span>Studio</span></div>', unsafe_allow_html=True)
    nav = st.radio("Navigate", ["📊 Overview", "🔮 Predict"], label_visibility="collapsed")
    st.markdown("##### Filters")
    date_range = st.selectbox("Date range", ["All time", "Last 90 days", "Last 30 days"])
    species_filter = st.multiselect("Species", classes, default=classes)
    st.caption("Dates are simulated for this demo.")

# Header row: page title, search, live sync toggle
title_col, search_col, live_col = st.columns([3, 3, 2])
with title_col:
    st.markdown(f"## {'Overview' if nav.startswith('📊') else 'Predict'}")
with search_col:
    search = st.text_input("Search", placeholder="Search species or flower ID", label_visibility="collapsed")
with live_col:
    live = st.toggle("Live data sync", value=True)
    st.caption("● Live" if live else "○ Paused")

# Apply sidebar filters
f = df[df["Species"].isin(species_filter)]
if date_range != "All time":
    days = 90 if date_range == "Last 90 days" else 30
    f = f[f["Date"] >= df["Date"].max() - pd.Timedelta(days=days)]

if nav.startswith("📊"):
    render_overview(df, f, metrics, search)
else:
    render_predict(df, model, scaler, classes)
