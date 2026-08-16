# 🚗 Ford Car Price Prediction Dashboard

![Ford Logo](https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_logo_flat.svg)

## 📌 Project Overview
This project predicts **Ford car prices** using Machine Learning models and provides an **interactive Streamlit dashboard** with premium visualizations.

- Dataset: `ford.csv` (Ford car listings with features like year, mileage, tax, mpg, engineSize, model, transmission, fuelType, and price)
- Models used:
  - Linear Regression
  - Random Forest
  - XGBoost
- Dashboard built with **Streamlit + Plotly** for interactive analysis.

---

## ⚙️ Features
- 📊 **Model Performance Comparison** (R² Score & RMSE)
- 🔎 **Feature Importance** (Random Forest & XGBoost)
- 📈 **Actual vs Predicted Prices** (Scatter Plot)
- 📉 **Residuals Distribution** (Error Analysis)
- 🎛️ **Interactive Sidebar Filters**
  - Car Model
  - Fuel Type
  - Transmission
  - Year Range
  - Mileage Range
- 🖱️ **Hover Tooltips** → See detailed info when hovering over data points
- 💾 **Download Option** → Export filtered dataset or predictions

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, plotly)
- **Streamlit** (for interactive dashboard)
- **GitHub** (for version control & portfolio showcase)

---
Install dependencies:

bash
pip install -r requirements.txt
Run Streamlit app:

bash
streamlit run app.py
Open browser at:

Code
http://localhost:8501
📊 Sample Results
Model	R² Score	RMSE
Linear Regression	0.705	2569
Random Forest	0.926	1289
XGBoost	0.937	1193


👉 XGBoost performed best with highest accuracy and lowest error.

✨ Screenshots
(Add screenshots of your dashboard here once deployed)

🌐 Deployment
You can deploy this app on:

Streamlit Cloud

Heroku

Render

👨‍💻 Author
Vishesh Tyagi  
Aspiring Data Analyst | Python • ML • Streamlit •
## 🚀 How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/ford-price-prediction.git
   cd ford-price-prediction
