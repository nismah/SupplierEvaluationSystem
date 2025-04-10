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

# Shift normalized scores to match model output range
X = df[criteria]
y = df['Score']
score_min = y.min()
score_shift = abs(score_min) if score_min < 0 else 0
y_shifted = y + score_shift

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y_shifted)

st.title("🤖 AI-Driven Supplier Evaluation System")
st.markdown("""
This system evaluates and ranks suppliers based on multiple criteria using a machine learning model. 

**Score Meaning:** Higher scores represent better suppliers. The score is derived from weighted inputs across Price, Delivery, Quality, Service, and Flexibility.

- Lower Price and Delivery times contribute positively (via negative weights).
- Higher Quality, Service, and Flexibility also increase the score.
- All scores are normalized and shifted to ensure they remain **positive and comparable**.
""")

# Tabs for Ranking and Manual Evaluation
tab1, tab2 = st.tabs(["📊 Supplier Ranking", "✍️ Manual Evaluation"])

with tab1:
    st.subheader("Set Weights for Each Criterion")
    price_weight = st.slider('💰 Price Weight', -1.0, 1.0, -0.3, help="Lower is better")
    delivery_weight = st.slider('🚚 Delivery Weight', -1.0, 1.0, -0.15, help="Faster is better")
    quality_weight = st.slider('📦 Quality Weight', -1.0, 1.0, 0.25, help="Higher is better")
    service_weight = st.slider('🔧 Service Weight', -1.0, 1.0, 0.2, help="Higher is better")
    flexibility_weight = st.slider('🤹 Flexibility Weight', -1.0, 1.0, 0.1, help="Higher is better")

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
        ranked_df['WeightedScore'] += abs(min_score) if min_score < 0 else 0
        # Normalize to 0-100
        ranked_df['WeightedScore'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(ranked_df[['WeightedScore']])
        ranked_df = ranked_df.sort_values(by='WeightedScore', ascending=False).reset_index(drop=True)
        st.dataframe(ranked_df[['Supplier', 'WeightedScore'] + criteria], use_container_width=True)

        # Chart: Top 10 Suppliers
        st.subheader("📈 Top 10 Suppliers")
        fig, ax = plt.subplots()
        ax.barh(ranked_df['Supplier'][:10][::-1], ranked_df['WeightedScore'][:10][::-1])
        ax.set_xlabel("Score (0–100)")
        ax.set_title("Top 10 Suppliers Based on Current Weights")
        st.pyplot(fig)

with tab2:
    st.subheader("Evaluate a New Supplier")
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
        predicted = model.predict(manual_input)[0] - score_shift
        predicted_norm = MinMaxScaler(feature_range=(0, 100)).fit(y.values.reshape(-1, 1)).transform([[predicted]])[0][0]
        st.success(f"Predicted Supplier Score (0–100): {predicted_norm:.2f}")

        # Compare with top suppliers visually
        if 'ranked_df' in locals():
            comparison_df = ranked_df[['Supplier', 'WeightedScore']].head(5).copy()
            comparison_df.loc[len(comparison_df)] = ['Your Supplier', predicted_norm]
            comparison_df = comparison_df.sort_values(by='WeightedScore', ascending=False)

            st.subheader("📊 Comparison with Top 5 Suppliers")
            fig2, ax2 = plt.subplots()
            ax2.barh(comparison_df['Supplier'][::-1], comparison_df['WeightedScore'][::-1], color=["skyblue" if s != 'Your Supplier' else "orange" for s in comparison_df['Supplier'][::-1]])
            ax2.set_xlabel("Score (0–100)")
            ax2.set_title("Your Supplier vs Top 5")
            st.pyplot(fig2)
