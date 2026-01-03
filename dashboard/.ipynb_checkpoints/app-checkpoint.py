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

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Uber Ride Analytics & Demand Prediction",
    layout="wide"
)

# -------------------------------------------------
# Paths (absolute & safe)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "uber_demand_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "models", "feature_columns.pkl")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "uber_processed.csv")

# -------------------------------------------------
# Load model & feature columns
# -------------------------------------------------
@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_feature_columns():
    return joblib.load(FEATURES_PATH)

try:
    model = load_model()
    feature_columns = load_feature_columns()
except FileNotFoundError:
    st.error("❌ Model or feature columns not found. Run model training notebook first.")
    st.stop()

# -------------------------------------------------
# Load processed data (for visuals)
# -------------------------------------------------
if os.path.exists(DATA_PATH):
    data = pd.read_csv(DATA_PATH)
else:
    data = None
    st.warning("Processed data not found. Visuals disabled.")

# -------------------------------------------------
# Prediction function (CORRECT one-hot logic)
# -------------------------------------------------
def predict_demand(hour, day_of_week, month, base):
    # Create empty row with all features = 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Fill numeric features
    input_df["hour"] = hour
    input_df["day_of_week"] = day_of_week
    input_df["month"] = month

    # Activate correct Base column
    base_col = f"Base_{base}"
    if base_col in input_df.columns:
        input_df[base_col] = 1

    prediction = model.predict(input_df)
    return int(prediction[0])

# -------------------------------------------------
# Sidebar inputs
# -------------------------------------------------
st.sidebar.header("🚕 Ride Demand Inputs")

hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0,1,2,3,4,5,6],
    format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
)
month = st.sidebar.slider("Month", 1, 12, 9)

# Extract Base options from feature columns
base_options = sorted(
    col.replace("Base_", "") 
    for col in feature_columns 
    if col.startswith("Base_")
)

base = st.sidebar.selectbox("Uber Base Zone", base_options)

# -------------------------------------------------
# Prediction output
# -------------------------------------------------
st.title("Uber Ride Analytics & Demand Prediction")

if st.sidebar.button("Predict Demand"):
    demand = predict_demand(hour, day_of_week, month, base)

    st.success(f"""
    🚕 **Predicted Ride Demand**
    
    • **Base:** {base}  
    • **Time:** {hour}:00  
    • **Day:** {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_of_week]}  
    • **Month:** {month}  

    ### 📈 **Estimated Rides: {demand}**
    """)

# -------------------------------------------------
# Dashboard visuals
# -------------------------------------------------
if data is not None:

    st.markdown("---")
    st.header("📊 Historical Demand Insights")

    col1, col2 = st.columns(2)

    # Hourly distribution
    with col1:
        st.subheader("Rides by Hour")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x="hour", data=data, ax=ax)
        ax.set_title("Total Pickups per Hour")
        st.pyplot(fig)

    # Day of week distribution
    with col2:
        st.subheader("Rides by Day of Week")
        fig2, ax2 = plt.subplots(figsize=(8,4))
        sns.countplot(x="day_of_week", data=data, ax=ax2)
        ax2.set_title("Total Pickups per Day")
        st.pyplot(fig2)

    # Base-wise demand
    st.subheader("🔥 Hot Zones (Total Rides per Base)")
    base_counts = data["Base"].value_counts().reset_index()
    base_counts.columns = ["Base", "Total Rides"]

    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.barplot(data=base_counts, x="Base", y="Total Rides", ax=ax3)
    ax3.set_title("Demand by Base Zone")
    st.pyplot(fig3)

    # Hourly trend for selected base
    st.subheader(f"📈 Hourly Trend for Base {base}")
    base_data = data[data["Base"] == base]

    if not base_data.empty:
        hourly = base_data["hour"].value_counts().sort_index()
        fig4, ax4 = plt.subplots(figsize=(10,4))
        sns.lineplot(x=hourly.index, y=hourly.values, marker="o", ax=ax4)
        ax4.set_xlabel("Hour")
        ax4.set_ylabel("Ride Count")
        st.pyplot(fig4)

# -------------------------------------------------
# Feature importance
# -------------------------------------------------
st.markdown("---")
st.header("🧠 Model Feature Importance")

importance = model.feature_importances_
fi_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig5, ax5 = plt.subplots(figsize=(10,5))
sns.barplot(data=fi_df.head(15), x="Importance", y="Feature", ax=ax5)
ax5.set_title("Top Influencing Features")
st.pyplot(fig5)
