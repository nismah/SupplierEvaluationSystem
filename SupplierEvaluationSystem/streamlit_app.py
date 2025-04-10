# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data from predefined file
df = pd.read_csv("SupplierEvaluationSystem/updated_dummy_supplier_data.csv")

# Page config and title
st.set_page_config(page_title="Supplier Evaluation Portal", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #f5f7fa;}
        h1 {color: #006d77;}
        .css-1d391kg {background-color: #edf6f9 !important; border-radius: 10px; padding: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🔍 AI-Driven Supplier Evaluation Portal")
st.markdown("Adjust the weights below to prioritize what's most important to you:")

# Sidebar sliders
with st.sidebar:
    st.header("⚙️ Set Evaluation Weights")
    price_weight = st.slider('💰 Price (lower is better)', -1.0, 1.0, -0.3)
    delivery_weight = st.slider('🚚 Delivery (lower is better)', -1.0, 1.0, -0.15)
    quality_weight = st.slider('📦 Quality', -1.0, 1.0, 0.25)
    service_weight = st.slider('🔧 Service', -1.0, 1.0, 0.2)
    flexibility_weight = st.slider('🤹 Flexibility', -1.0, 1.0, 0.1)
    go = st.button("🔄 Regenerate Results")

# Process when button is clicked
if go:
    criteria = ['Price', 'Delivery', 'Quality', 'Service', 'Flexibility']
    weights = {
        'Price': price_weight,
        'Delivery': delivery_weight,
        'Quality': quality_weight,
        'Service': service_weight,
        'Flexibility': flexibility_weight
    }

    norm_df = df.copy()
    norm_df[criteria] = MinMaxScaler().fit_transform(df[criteria])
    df['Score'] = sum(norm_df[c] * weights[c] for c in criteria)
    df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)

    st.success("🎯 Ranked Suppliers Based on Current Weights")
    st.dataframe(df_sorted[['Supplier', 'Score'] + criteria], use_container_width=True)

    st.download_button("⬇️ Download Results as CSV", df_sorted.to_csv(index=False), "ranked_suppliers.csv")
