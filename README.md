# Air Pollution Analysis using Machine Learning

## Project Overview
This project focuses on analyzing air pollution data and predicting air quality levels using Machine Learning techniques. It explores environmental factors such as PM2.5, PM10, NO2, SO2, CO, temperature, and humidity to understand their impact on air quality.

The project also includes data visualization and classification modeling to predict air quality categories such as **Good, Moderate, Poor, and Hazardous**.

---

## Project Features
- Air Quality Prediction using Machine Learning
- Exploratory Data Analysis (EDA)
- Data Visualization
- Model Comparison
- Interactive Streamlit Dashboard

---

## Dataset Information
- **Total Records:** 5000  
- **Total Features:** 10  
- **Target Variable:** Air Quality  

### Selected features:
- Temperature  
- Humidity  
- PM2.5  
- PM10  
- NO2  
- SO2  
- CO    
- Population Density  

### Target Classes:
Air Quality:
- Good  
- Moderate  
- Poor  
- Hazardous  

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Jupyter Notebook 
- Streamlit
- Pickle
- HTML
- CSS

---

## Project Workflow

```text
┌─────────┐
│ Dataset │
└────┬────┘
     ▼
┌───────────────┐
│ Data Cleaning │
└────┬──────────┘
     ▼
┌───────────────┐
│      EDA      │
└────┬──────────┘
     ▼
┌───────────────────┐
│ Feature Selection │
└────┬──────────────┘
     ▼
┌────────────────┐
│ Model Training │
└────┬───────────┘
     ▼
┌──────────────────┐
│ Model Evaluation │
└────┬─────────────┘
     ▼
┌──────────────────────┐
│ Model Serialization  │
│        (.pkl)        │
└────┬─────────────────┘
     ▼
┌─────────────────────┐
│ Streamlit Dashboard │
└─────────────────────┘
```

---

## Algorithms Used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

---

## Model Performance
| Model | Accuracy |
|--------|----------|
| Logistic Regression | 0.974 |
| Decision Tree | 0.950 |
| Random Forest | 0.971 |

---

## How to Run the Project
```bash
# Clone the repository
git clone https://github.com/kriti123680/Air-Pollution-Analysis.git

# Navigate to project folder
cd Air-Pollution-Analysis

# Install required libraries
pip install -r requirements.txt

# Run Jupyter Notebook
jupyter notebook

# Run the Streamlit dashboard
streamlit run notebook/dashboard.py
```
---
## Project Structure
```text
Air-Pollution-Analysis
│
├── dataset
├── notebook
├── README.md
├── requirements.txt
```

---
## Team Members
- Kriti Mathya
- Sanjaya Moktan
