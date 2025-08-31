import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide", page_title="YouTube Channel Analytics Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv('youtube_channel_real_performance_analytics.csv')

@st.cache_resource
def load_model():
    return joblib.load('youtube_revenue_predictor.pkl')

data = load_data()
model = load_model()

st.title("YouTube Channel Performance Dashboard")

st.sidebar.header("Filters")
min_views, max_views = int(data['Views'].min()), int(data['Views'].max())
views_range = st.sidebar.slider("Views Range", min_views, max_views, (min_views, max_views))

min_date = data['Video Publish Time'].min()
max_date = data['Video Publish Time'].max()
date_range = st.sidebar.date_input("Publish Date Range", [min_date, max_date])

filtered = data[
    (data['Views'] >= views_range[0]) & (data['Views'] <= views_range[1]) &
    (pd.to_datetime(data['Video Publish Time']) >= pd.to_datetime(date_range[0])) &
    (pd.to_datetime(data['Video Publish Time']) <= pd.to_datetime(date_range[1]))
]

st.subheader("Filtered Data Overview")
st.dataframe(filtered.head(50))

st.subheader("Revenue Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(filtered['Estimated Revenue (USD)'], bins=40, color='g', kde=True, ax=ax1)
st.pyplot(fig1)

st.subheader("Top 10 Videos by Revenue")
top10 = filtered.sort_values(by="Estimated Revenue (USD)", ascending=False).head(10)
st.dataframe(top10[['Estimated Revenue (USD)', 'Views', 'Subscribers']])

st.subheader("Feature Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10,7))
sns.heatmap(filtered.corr(numeric_only=True), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
st.pyplot(fig2)

st.subheader("Revenue Over Time")
revenue_time = filtered.groupby('Video Publish Time')['Estimated Revenue (USD)'].sum().reset_index()
fig3 = px.line(revenue_time, x='Video Publish Time', y='Estimated Revenue (USD)', title='Total Revenue Over Time')
st.plotly_chart(fig3)

st.subheader("Model Evaluation Metrics")
mse, rmse, mae, r2 = 0.0, 0.0, 0.0, 0.0  

st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"R2 Score: {r2:.2f}")

st.subheader("Predict Revenue for Custom Metrics")
views = st.number_input("Views", min_value=0, value=int(data['Views'].median()))
subscribers = st.number_input("Subscribers", min_value=0, value=int(data['Subscribers'].median()))
likes = st.number_input("Likes", min_value=0, value=int(data['Likes'].median()))
shares = st.number_input("Shares", min_value=0, value=int(data['Shares'].median()))
comments = st.number_input("Comments", min_value=0, value=int(data['New Comments'].median()))
engagement_rate = st.number_input("Engagement Rate (%)", min_value=0.0, value=float(data['Like Rate (%)'].median()))

if st.button("Predict Revenue"):
    features = np.array([[views, subscribers, likes, shares, comments, engagement_rate,
                          0, 0, 0, 0]])  
    predicted = model.predict(features)[0]
    st.success(f"Predicted Estimated Revenue (USD): ${predicted:,.2f}")

if st.button("Download Filtered Data as CSV"):
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV", data=csv, file_name='filtered_youtube_data.csv')
 