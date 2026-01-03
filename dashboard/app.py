# # dashboard/app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# st.set_page_config(page_title="Uber Ride Demand Prediction", layout="wide")

# # -------------------------------
# # 1️⃣ Define paths
# # -------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, '../models/uber_demand_model.pkl')
# ENCODER_PATH = os.path.join(BASE_DIR, '../models/base_encoder.pkl')

# # -------------------------------
# # 2️⃣ Load model & encoder
# # -------------------------------
# @st.cache_data
# def load_model():
#     return joblib.load(MODEL_PATH)

# @st.cache_data
# def load_le_encoder():
#     return joblib.load(ENCODER_PATH)

# try:
#     model = load_model()
#     le = load_le_encoder()
# except FileNotFoundError:
#     st.error("Model or encoder file not found in ../models folder.")
#     st.stop()

# # -------------------------------
# # 3️⃣ Prediction function
# # -------------------------------
# def predict_demand(model, hour, day_of_week, month, base):
#     # Encode Base using loaded LabelEncoder
#     base_encoded = le.transform([base])[0]
#     X = np.array([[hour, day_of_week, month, base_encoded]])
#     pred = model.predict(X)
#     return int(pred[0])

# # -------------------------------
# # 4️⃣ Sidebar Inputs
# # -------------------------------
# st.sidebar.header("Input Parameters")
# hour = st.sidebar.slider("Hour of the day", 0, 23, 12)
# day_of_week = st.sidebar.slider("Day of the week (0=Mon,6=Sun)", 0, 6, 2)
# month = st.sidebar.slider("Month (1-12)", 1, 12, 9)

# # Base options from encoder
# base_options = list(le.classes_)
# base = st.sidebar.selectbox("Select Base Zone", base_options)

# # -------------------------------
# # 5️⃣ Predict button
# # -------------------------------
# if st.sidebar.button("Predict Ride Demand"):
#     predicted_rides = predict_demand(model, hour, day_of_week, month, base)
#     st.success(f"🚕 Predicted Rides for Base {base} at {hour}:00 on day {day_of_week}, month {month}: **{predicted_rides} rides**")

# # -------------------------------
# # 6️⃣ Optional Visualizations
# # -------------------------------
# st.header("Uber Ride Demand Insights")

# # Load processed data for visuals
# processed_data_path = os.path.join(BASE_DIR, '../data/processed/uber_processed.csv')
# if os.path.exists(processed_data_path):
#     data = pd.read_csv(processed_data_path)

#     st.subheader("Ride Distribution by Hour")
#     fig, ax = plt.subplots(figsize=(10,5))
#     sns.countplot(x='hour', data=data, ax=ax)
#     ax.set_title("Total Uber Pickups by Hour")
#     st.pyplot(fig)

#     st.subheader("Ride Distribution by Day of Week")
#     fig2, ax2 = plt.subplots(figsize=(10,5))
#     sns.countplot(x='day_of_week', data=data, ax=ax2)
#     ax2.set_title("Total Uber Pickups by Day of Week")
#     st.pyplot(fig2)

# else:
#     st.warning("Processed data not found at ../data/processed/uber_processed.csv. Visuals cannot be generated.")
 #





#second code:

# dashboard/app.py
# 



# third code:
# dashboard/app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ----------------------------------
# # Page config
# # ----------------------------------
# st.set_page_config(
#     page_title="Uber Ride Demand Analytics",
#     layout="wide"
# )

# # ----------------------------------
# # Paths
# # ----------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "../models/uber_demand_model.pkl")
# DATA_PATH = os.path.join(BASE_DIR, "../data/processed/uber_processed.csv")

# # ----------------------------------
# # Load model
# # ----------------------------------
# @st.cache_resource
# def load_model():
#     return joblib.load(MODEL_PATH)

# if not os.path.exists(MODEL_PATH):
#     st.error("❌ Model file not found")
#     st.stop()

# model = load_model()

# # ----------------------------------
# # Load data
# # ----------------------------------
# if not os.path.exists(DATA_PATH):
#     st.error("❌ Processed data not found")
#     st.stop()

# data = pd.read_csv(DATA_PATH)

# # ----------------------------------
# # Sidebar Inputs
# # ----------------------------------
# st.sidebar.title("🚕 Ride Demand Prediction")

# hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
# day_of_week = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
# month = st.sidebar.slider("Month", 1, 12, 9)

# base = st.sidebar.selectbox(
#     "Base Zone",
#     sorted(data["Base"].unique())
# )

# # ----------------------------------
# # Prediction logic (One-Hot safe)
# # ----------------------------------
# def predict_demand(hour, day_of_week, month, base):
#     input_df = pd.DataFrame({
#         "hour": [hour],
#         "day_of_week": [day_of_week],
#         "month": [month],
#         "Base": [base]
#     })

#     input_encoded = pd.get_dummies(input_df, columns=["Base"], drop_first=True)
#     input_encoded = input_encoded.reindex(
#         columns=model.feature_names_in_,
#         fill_value=0
#     )

#     return int(model.predict(input_encoded)[0])

# # ----------------------------------
# # Predict button
# # ----------------------------------
# if st.sidebar.button("Predict Demand"):
#     rides = predict_demand(hour, day_of_week, month, base)
#     st.sidebar.success(f"Predicted Rides: {rides}")

# # ----------------------------------
# # Dashboard Title
# # ----------------------------------
# st.title("📊 Uber Ride Demand Analytics Dashboard")

# # ----------------------------------
# # KPI Metrics
# # ----------------------------------
# total_rides = len(data)
# avg_rides_per_hour = round(total_rides / 24, 2)
# peak_hour = data["hour"].value_counts().idxmax()

# col1, col2, col3 = st.columns(3)
# col1.metric("Total Rides", f"{total_rides:,}")
# col2.metric("Avg Rides / Hour", avg_rides_per_hour)
# col3.metric("Peak Hour", f"{peak_hour}:00")

# st.divider()

# # ----------------------------------
# # Trends
# # ----------------------------------
# col4, col5 = st.columns(2)

# with col4:
#     st.subheader("Hourly Ride Trend")
#     hourly = data.groupby("hour").size()
#     fig, ax = plt.subplots(figsize=(8,4))
#     hourly.plot(marker="o", ax=ax)
#     ax.set_xlabel("Hour")
#     ax.set_ylabel("Ride Count")
#     st.pyplot(fig)

# with col5:
#     st.subheader("Day-wise Ride Trend")
#     daywise = data.groupby("day_of_week").size()
#     fig, ax = plt.subplots(figsize=(8,4))
#     sns.barplot(x=daywise.index, y=daywise.values, ax=ax)
#     ax.set_xlabel("Day of Week")
#     ax.set_ylabel("Ride Count")
#     st.pyplot(fig)

# st.divider()

# # ----------------------------------
# # Hot Zones
# # ----------------------------------
# st.subheader("🔥 Hot Zones by Base")
# base_counts = data["Base"].value_counts().reset_index()
# base_counts.columns = ["Base", "Total Rides"]

# fig, ax = plt.subplots(figsize=(10,4))
# sns.barplot(x="Base", y="Total Rides", data=base_counts, ax=ax)
# st.pyplot(fig)
# # === ADD THIS SECTION BELOW HOT ZONES ===

# st.subheader("🗺️ Live Demand Heatmap (Pickup Density)")

# # Ensure latitude/longitude columns exist
# lat_cols = [c for c in data.columns if c.lower() in ["lat", "latitude"]]
# lon_cols = [c for c in data.columns if c.lower() in ["lon", "lng", "longitude"]]

# if lat_cols and lon_cols:
#     heat_df = data[[lat_cols[0], lon_cols[0]]].dropna()
#     heat_df.columns = ["lat", "lon"]

#     # Optional sampling for performance
#     if len(heat_df) > 50000:
#         heat_df = heat_df.sample(50000, random_state=42)

#     st.map(heat_df)
# else:
#     st.info("Latitude/Longitude columns not found for heatmap.")

# # ----------------------------------
# # Base-specific hourly trend
# # ----------------------------------
# st.subheader(f"📍 Hourly Demand – Base {base}")
# base_data = data[data["Base"] == base]

# if not base_data.empty:
#     hourly_base = base_data.groupby("hour").size()
#     fig, ax = plt.subplots(figsize=(10,4))
#     hourly_base.plot(marker="o", ax=ax)
#     ax.set_xlabel("Hour")
#     ax.set_ylabel("Ride Count")
#     st.pyplot(fig)

# st.divider()

# # ----------------------------------
# # Feature Importance
# # ----------------------------------
# st.subheader("🧠 Model Feature Importance")

# fi_df = pd.DataFrame({
#     "Feature": model.feature_names_in_,
#     "Importance": model.feature_importances_
# }).sort_values(by="Importance", ascending=False)

# fig, ax = plt.subplots(figsize=(10,4))
# sns.barplot(x="Importance", y="Feature", data=fi_df, ax=ax)
# st.pyplot(fig)


#fourth code:  


# dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------
# Page Config
# ----------------------------------
st.set_page_config(
    page_title="Uber Ride Demand Prediction",
    layout="wide"
)

# ----------------------------------
# Paths
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "uber_demand_model.pkl"))
ENCODER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "base_encoder.pkl"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "raw", "uber-raw-data-sep14.csv"))

# ----------------------------------
# Load model & encoder
# ----------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_encoder():
    return joblib.load(ENCODER_PATH)

try:
    model = load_model()
    encoder = load_encoder()
except Exception:
    st.error("❌ Model or encoder not found. Please train the model first.")
    st.stop()

# ----------------------------------
# Prediction Function
# ----------------------------------
def predict_demand(hour, day_of_week, month, base):
    base_encoded = encoder.transform([base])[0]
    X = np.array([[hour, day_of_week, month, base_encoded]])
    prediction = model.predict(X)[0]
    return int(prediction)

# ----------------------------------
# Sidebar Inputs
# ----------------------------------
st.sidebar.title("🚕 Ride Inputs")

hour = st.sidebar.slider("Hour", 0, 23, 12)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x]
)

month = st.sidebar.selectbox("Month", list(range(1, 13)))

base = st.sidebar.selectbox("Base Zone", list(encoder.classes_))

# ----------------------------------
# Main Prediction + What-if Simulation
# ----------------------------------
st.title("🚖 Uber Ride Demand Dashboard")

if st.sidebar.button("🔮 Predict Demand"):
    predicted_rides = predict_demand(hour, day_of_week, month, base)

    # --- WHAT-IF SIMULATION ---
    surge_percent = 0.20
    rides_per_driver = 10
    extra_drivers = int((predicted_rides * surge_percent) / rides_per_driver)

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Rides", predicted_rides)
    col2.metric("Demand Surge", "20%")
    col3.metric("Extra Drivers Needed", extra_drivers)

    st.info(
        f"""
        **Operational Insight:**  
        If demand increases by **20%**, approximately **{extra_drivers} additional drivers**
        will be required in **Base {base}** at **{hour}:00**.
        """
    )

# ----------------------------------
# Load data for visuals
# ----------------------------------
if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    df['hour'] = df['Date/Time'].dt.hour
    df['day_of_week'] = df['Date/Time'].dt.dayofweek
else:
    df = None

# ----------------------------------
# Visual Analytics
# ----------------------------------
if df is not None:
    st.markdown("---")
    st.subheader("📊 Ride Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Hourly Demand")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.countplot(x="hour", data=df, ax=ax1)
        ax1.set_xlabel("Hour")
        ax1.set_ylabel("Ride Count")
        st.pyplot(fig1)

    with col2:
        st.markdown("### Weekly Demand")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.countplot(x="day_of_week", data=df, ax=ax2)
        ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax2.set_xlabel("Day")
        ax2.set_ylabel("Ride Count")
        st.pyplot(fig2)

    st.markdown("### 🔥 Demand by Base Zone")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sns.countplot(x="Base", data=df, ax=ax3)
    ax3.set_xlabel("Base Zone")
    ax3.set_ylabel("Ride Count")
    st.pyplot(fig3)

    st.markdown(f"### ⏱ Hourly Trend for Base {base}")
    base_df = df[df["Base"] == base]
    hourly = base_df.groupby("hour").size()

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    hourly.plot(kind="line", marker="o", ax=ax4)
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Ride Count")
    st.pyplot(fig4)

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.caption("Uber Ride Demand Prediction | ML + Analytics + What-If Simulation")
