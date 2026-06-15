import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

model = pickle.load(open("model.pkl", "rb"))

df = pd.read_csv("updated_pollution_dataset.csv")

st.title("🌫 Air Pollution Analysis Dashboard")
menu = st.sidebar.selectbox(
    "Choose Section",
    ["Prediction", "Data Overview", "Visualizations"]
)
if menu == "Prediction":

    st.subheader("Predict Air Quality")

    temperature = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    pm25 = st.number_input("PM2.5")
    pm10 = st.number_input("PM10")
    no2 = st.number_input("NO2")
    so2 = st.number_input("SO2")
    co = st.number_input("CO")
    proximity = st.number_input("Proximity to Industrial Areas")
    population = st.number_input("Population Density")

    if st.button("Predict"):

        input_data = np.array([[
            temperature, humidity, pm25, pm10,
            no2, so2, co, proximity, population
        ]])

        prediction = model.predict(input_data)

        labels = ["Good", "Moderate", "Poor", "Hazardous"]
        result = labels[prediction[0]]

        st.success(f"Predicted Air Quality: {result}")

elif menu == "Data Overview":

    st.subheader("Dataset Overview")

    st.write(df.head())

    st.write("Shape:", df.shape)

    st.write("Missing values:")
    st.write(df.isnull().sum())

elif menu == "Visualizations":

    st.subheader("Air Quality Distribution")

    fig1 = plt.figure()
    df["Air Quality"].value_counts().plot(kind="bar")
    plt.title("Air Quality Count")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")

    corr = df.drop("Air Quality", axis=1).corr()

    fig2 = plt.figure()
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.title("Feature Correlation")
    st.pyplot(fig2)

    st.subheader("PM2.5 Distribution")

    fig3 = plt.figure()
    df["PM2.5"].hist(bins=20)
    plt.title("PM2.5 Distribution")
    st.pyplot(fig3)