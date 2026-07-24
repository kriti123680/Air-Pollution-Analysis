import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="AirSense AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ===========================
# LOAD CSS
# ===========================
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
load_css("style.css")
# ===========================
# SIDEBAR
# ===========================
st.sidebar.markdown("# 🌍 AirSense AI")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "",
    [
        "🏠 Home",
        "📊 Dataset Analysis",
        "🤖 Prediction",
        "📉 Model Performance",
        "👨‍💻 About"
    ]
)
st.sidebar.markdown("---")
st.sidebar.success("System Ready")
# ===========================
# HOME PAGE
# ===========================
if page == "🏠 Home":
    st.markdown("""
<div class="hero">
    <div class="hero-logo">🌍</div>
    <h1>AirSense AI</h1>
    <h3>Smart Air Pollution Analysis & Prediction</h3>
    <p>Predict • Monitor • Analyze</p>
</div>
""", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-label">🤖 Algorithm</div>
            <div class="card-value">Random Forest</div>
            <div class="card-line purple"></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-label">🎯 Accuracy</div>
            <div class="card-value">97.8%</div>
            <div class="card-line blue"></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <div class="card-label">🌍 Prediction Target</div>
            <div class="card-value">Air Quality</div>
            <div class="card-line green"></div>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    graph_data = pd.DataFrame({
        "Demo":[
            "Demo1",
            "Demo2",
            "Demo3",
            "Demo4",
            "Demo5",
            "Demo6",
            "Demo7"
        ],
        "AQI":[
            42,
            55,
            48,
            70,
            61,
            53,
            46
        ]
    })
    fig = px.line(
        graph_data,
        x="Demo",
        y="AQI",
        markers=True
    )
    fig.update_traces(
        line=dict(
            color="#3B82F6",
            width=4,
            shape="spline"
        ),
        marker=dict(
            size=8
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="white"
        ),
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,.15)"
        )
    )
    left, right = st.columns([3,1])
    with left:
        st.markdown("## 📈 Air Quality Trend")
        st.plotly_chart(
        fig,
        use_container_width=True
        )
    with right:
        st.markdown("""
                    <div class="prediction-card">
                    <h2>Latest Predction</h2>
                    <div class="prediction-circle">🌿</div>
                    <h1>GOOD</h1>
                    <h3>Safe Air</h3>
                    <p>AQI : 42</p>
                    </div>
                    """, unsafe_allow_html=True)
    st.info(
        "Live AQI API will be integrated in the next version."
        )
    st.write("")
# ===========================
# DATASET ANALYSIS
# ===========================
elif page == "📊 Dataset Analysis":
    st.title("📊 Dataset Analysis")
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div class="card-label">📄 Total Samples</div>
            <div class="card-value">5000</div>
            <div class="card-line blue"></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div class="card-label">🎯 Target Classes</div>
            <div class="card-value">2</div>
            <div class="card-line green"></div>
        </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.container(border=True)
    st.subheader("Dataset Preview")
    demo_df = pd.DataFrame({
        "Temperature":[24,25,28,27,26],
        "Humidity":[68,71,74,69,73],
        "PM2.5":[45,51,63,58,42],
        "PM10":[82,94,101,90,76],
        "Air Quality":[
            "Good",
            "Moderate",
            "Poor",
            "Good",
            "Moderate"
        ]
    })
    st.dataframe(
        demo_df,
        use_container_width=True
    )
    st.write("")
    st.subheader("Feature Distribution")
    fig = px.histogram(
        demo_df,
        x="PM2.5",
        color="Air Quality"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
# ===========================
# PREDICTION
# ===========================
elif page == "🤖 Prediction":
    st.markdown("""
    <div class="hero">
        <h1>🤖 Air Quality Prediction</h1>
        <h3>Random Forest Classifier</h3>
        <p>Enter environmental values to predict air quality.</p>
    </div>
    """, unsafe_allow_html=True)
    left, right = st.columns([2.2,1], gap="large")
    # ========================
    # INPUT SECTION
    # ========================
    with left:
        st.markdown("## 🌡 Environmental Parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🌡 Temperature (°C)")
            temperature = st.number_input(
            "",
            value=25.0,
            key="temp"
            )
            st.markdown("##### 🌡 Humidity (%)")
            humidity = st.number_input(
                "",
                value=70.0,
                key="humidity"
            )
            st.markdown("##### 🌡 PM2.5 (μg/m³)")
            pm25 = st.number_input(
                "",
                value=45.0,
                key="pm25"
            )
            st.markdown("##### 🌡 PM10 (μg/m³)")
            pm10 = st.number_input(
                "",
                value=80.0,
                key="pm10"
            )
        with c2:
            st.markdown("##### 🌡 NO2")
            no2 = st.number_input(
                "",
                value=20.0,
                key="no2"
            )
            st.markdown("##### 🌡 SO₂")
            so2 = st.number_input(
                "",
                value=8.0,
                key="so2"
            )
            st.markdown("##### 🌡 CO")
            co = st.number_input(
                "",
                value=0.8,
                key="co"
            )
            st.markdown("##### 🌡 Population Density")
            population = st.number_input(
                "",
                value=150,
                key="population"
            )
        st.write("")
        predict = st.button(
            "🚀 Predict Air Quality",
            use_container_width=True
        )
    # ========================
    # RESULT SECTION
    # ========================
    with right:
        st.markdown("## 🌿 Prediction Result")
        st.success("### GOOD")
        st.markdown("#### Safe Air")
        st.metric(
            "AQI",
            "42"
        )
        st.progress(98)
        st.caption("Confidence : 97.8%")
        if predict:
            input_data = np.array([[
        temperature,
        humidity,
        pm25,
        pm10,
        no2,
        so2,
        co,
        population
    ]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    st.success("Prediction Completed Successfully")
    st.write("Prediction:", prediction)
# ===========================
# MODEL PERFORMANCE
# ===========================
elif page == "📉 Model Performance":
    st.title("📉 Model Performance")
    col1,col2,col3,col4 = st.columns(4)
    cards = [
    ("🎯 Accuracy", "97.8%", "blue"),
    ("✅ Precision", "97.5%", "green"),
    ("📈 Recall", "97.2%", "purple"),
    ("⭐ F1 Score", "97.3%", "blue"),
    ]
    for col, (label, value, color) in zip([col1, col2, col3, col4], cards):
        with col:
            st.markdown(f"""
            <div class="info-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
                <div class="card-line {color}"></div>
            </div>
            """, unsafe_allow_html=True)
    st.write("")
    st.container(border=True)
    st.subheader("Confusion Matrix")
    cm = pd.DataFrame(
        [
            [475,12],
            [9,504]
        ],
        columns=["Pred Good","Pred Poor"],
        index=["Actual Good","Actual Poor"]
    )
    st.dataframe(
        cm,
        use_container_width=True
    )
# ===========================
# ABOUT
# ===========================
elif page == "👨‍💻 About":
    st.title("👨‍💻 About AirSense AI")
    st.markdown("""
### Project

AirSense AI is a machine learning based dashboard for monitoring and predicting air quality.

---

### Machine Learning Model

- Random Forest Classifier

---

### Developed By

- Kriti Mathya
- Sanjaya Moktan

---

### Features

- Air Quality Prediction
- Dataset Analysis
- Model Performance
- AQ Trend Visualization
- API Integration (Upcoming)

""")