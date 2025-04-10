# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data
st.set_page_config(page_title="Supplier Evaluation System", layout="wide")
df = pd.read_csv("updated_dummy_supplier_data.csv")

# Define evaluation criteria
criteria = ['Price', 'Delivery', 'Quality', 'Service', 'Flexibility']

# Compute Score if not already present
if 'Score' not in df.columns:
    weights = {'Price': -0.3, 'Delivery': -0.15, 'Quality': 0.25, 'Service': 0.2, 'Flexibility': 0.1}
    df['Score'] = (
        df['Price'] * weights['Price'] +
        df['Delivery'] * weights['Delivery'] +
        df['Quality'] * weights['Quality'] +
        df['Service'] * weights['Service'] +
        df['Flexibility'] * weights['Flexibility']
    )
    # Shift score to make all values positive
    min_score = df['Score'].min()
    df['Score'] += abs(min_score)

# Normalize data
scaler = MinMaxScaler()
df[criteria] = scaler.fit_transform(df[criteria])

# Train model
X = df[criteria]
y = df['Score']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

st.title("🤖 AI-Driven Supplier Evaluation System")

# Section 1: Best Supplier from Dataset with Weight Adjustment
st.header("📊 Weighted Supplier Ranking")
st.subheader("Set Weights for Each Criterion")
price_weight = st.slider('💰 Price Weight', -1.0, 1.0, -0.3)
delivery_weight = st.slider('🚚 Delivery Weight', -1.0, 1.0, -0.15)
quality_weight = st.slider('📦 Quality Weight', -1.0, 1.0, 0.25)
service_weight = st.slider('🔧 Service Weight', -1.0, 1.0, 0.2)
flexibility_weight = st.slider('🤹 Flexibility Weight', -1.0, 1.0, 0.1)

if st.button("🔄 Recalculate Rankings"):
    ranked_df = df.copy()
    weight_dict = {
        'Price': price_weight,
        'Delivery': delivery_weight,
        'Quality': quality_weight,
        'Service': service_weight,
        'Flexibility': flexibility_weight
    }
    ranked_df['WeightedScore'] = sum(ranked_df[c] * w for c, w in weight_dict.items())
    # Shift scores to positive range
    min_score = ranked_df['WeightedScore'].min()
    ranked_df['WeightedScore'] += abs(min_score)
    ranked_df = ranked_df.sort_values(by='WeightedScore', ascending=False).reset_index(drop=True)
    st.dataframe(ranked_df[['Supplier', 'WeightedScore'] + criteria], use_container_width=True)

    # Chart: Top 10 Suppliers
    st.subheader("📈 Top 10 Suppliers")
    fig, ax = plt.subplots()
    ax.barh(ranked_df['Supplier'][:10][::-1], ranked_df['WeightedScore'][:10][::-1])
    ax.set_xlabel("Score")
    ax.set_title("Top 10 Suppliers Based on Current Weights")
    st.pyplot(fig)

# Section 2: Score Prediction for Manual Input
st.header("✍️ Evaluate a New Supplier Manually")
price = st.slider("💰 Price", 0.0, 500.0, 250.0)
delivery = st.slider("🚚 Delivery Time (days)", 0.0, 30.0, 15.0)
quality = st.slider("📦 Quality Score", 0.0, 10.0, 5.0)
service = st.slider("🔧 Service Score", 0.0, 10.0, 5.0)
flexibility = st.slider("🤹 Flexibility Score", 0.0, 10.0, 5.0)

if st.button("🎯 Predict Supplier Score"):
    manual_input = pd.DataFrame([{
        'Price': price,
        'Delivery': delivery,
        'Quality': quality,
        'Service': service,
        'Flexibility': flexibility
    }])
    manual_input[criteria] = scaler.transform(manual_input[criteria])
    predicted = model.predict(manual_input)[0]
    st.success(f"Predicted Supplier Score: {predicted:.2f}")
