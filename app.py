import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import plot_importance

# -------------------------------
# Dashboard Title & Logo
# -------------------------------
st.set_page_config(page_title="Ford Car Price Prediction", layout="wide")

st.image("https://upload.wikimedia.org/wikipedia/commons/3/3e/Ford_logo_flat.svg", width=150)
st.title("🚗 Ford Car Price Prediction Dashboard")
st.markdown("### Machine Learning Models: Linear Regression, Random Forest, XGBoost")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("ford.csv")
df = df.dropna().drop_duplicates()

# Sidebar filter for car model
car_models = df['model'].unique()
selected_model = st.sidebar.selectbox("Select Car Model", car_models)

# Filter dataset by selected model
filtered_df = df[df['model'] == selected_model]

st.sidebar.write("📊 Selected Model Details")
st.sidebar.write(f"Total Records: {len(filtered_df)}")
st.sidebar.write(f"Average Price: £{filtered_df['price'].mean():,.0f}")
st.sidebar.write(f"Year Range: {filtered_df['year'].min()} - {filtered_df['year'].max()}")

# -------------------------------
# Encode categorical features
# -------------------------------
df['model'] = df['model'].astype('category').cat.codes
df['transmission'] = df['transmission'].astype('category').cat.codes
df['fuelType'] = df['fuelType'].astype('category').cat.codes

X = df[['year','mileage','tax','mpg','engineSize','model','transmission','fuelType']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Train Models
# -------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

xgb = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# -------------------------------
# Results Table
# -------------------------------
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "R2 Score": [r2_score(y_test, y_pred_lr),
                 r2_score(y_test, y_pred_rf),
                 r2_score(y_test, y_pred_xgb)],
    "RMSE": [np.sqrt(mean_squared_error(y_test, y_pred_lr)),
             np.sqrt(mean_squared_error(y_test, y_pred_rf)),
             np.sqrt(mean_squared_error(y_test, y_pred_xgb))]
})

st.subheader("📊 Model Performance Comparison")
st.dataframe(results.style.highlight_max(color="lightgreen", axis=0))

# -------------------------------
# Visualizations
# -------------------------------
st.subheader("🔎 Feature Importance (Random Forest)")
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=rf.feature_importances_, y=X.columns, ax=ax)
st.pyplot(fig)

st.subheader("🔎 Feature Importance (XGBoost)")
fig, ax = plt.subplots(figsize=(8,5))
plot_importance(xgb, max_num_features=10, importance_type='weight', ax=ax)
st.pyplot(fig)

st.subheader("📈 Actual vs Predicted Prices (XGBoost)")
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(y_test, y_pred_xgb, alpha=0.5, color="blue")
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Actual Price")
ax.set_ylabel("Predicted Price")
ax.set_title("XGBoost: Actual vs Predicted Car Prices")
st.pyplot(fig)

st.subheader("📉 Residuals Distribution (XGBoost)")
residuals = y_test - y_pred_xgb
fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True, color="purple", ax=ax)
ax.set_title("Residuals Distribution")
st.pyplot(fig)

st.success("✅ Dashboard Ready! Use sidebar to filter cars by model.")



import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("ford.csv")
df = df.dropna().drop_duplicates()

# Sidebar filter for car model
car_models = df['model'].unique()
selected_model = st.sidebar.selectbox(
    "Select Car Model", 
    car_models, 
    key="car_model_select"
)

# अगर दूसरा selectbox बनाना है तो अलग key दो
selected_fuel = st.sidebar.selectbox(
    "Select Fuel Type", 
    df['fuelType'].unique(), 
    key="fuel_type_select"
)


st.plotly_chart(fig, use_container_width=True)



hover_data=["year","mileage","engineSize","price","predicted_price"]

