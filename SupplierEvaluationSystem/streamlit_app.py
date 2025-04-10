# streamlit_app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("AI-Driven Supplier Evaluation")

# Upload CSV or input manually
uploaded_file = st.file_uploader("Upload supplier data CSV", type=["csv"])

# If file uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Define criteria weights (sliders for user input with default values)
    st.sidebar.header("Set Weights (adjust if needed)")
    price_weight = st.sidebar.slider('Price (lower is better)', -1.0, 1.0, -0.3)
    delivery_weight = st.sidebar.slider('Delivery (lower is better)', -1.0, 1.0, -0.15)
    quality_weight = st.sidebar.slider('Quality', -1.0, 1.0, 0.25)
    service_weight = st.sidebar.slider('Service', -1.0, 1.0, 0.2)
    flexibility_weight = st.sidebar.slider('Flexibility', -1.0, 1.0, 0.1)

    # Add button to regenerate results
    if st.button("Regenerate Results"):
        # Normalize relevant columns
        scaler = MinMaxScaler()
        norm_df = df.copy()
        criteria = ['Price', 'Delivery', 'Quality', 'Service', 'Flexibility']
        norm_df[criteria] = scaler.fit_transform(df[criteria])

        # Calculate final score based on weighted sum
        weights = {
            'Price': price_weight,
            'Delivery': delivery_weight,
            'Quality': quality_weight,
            'Service': service_weight,
            'Flexibility': flexibility_weight
        }

        df['Score'] = sum(norm_df[col] * weights[col] for col in criteria)
        df_sorted = df.sort_values('Score', ascending=False).reset_index(drop=True)

        # Display ranked suppliers
        st.success("\U0001F3AF Ranked Suppliers")
        st.dataframe(df_sorted[['Supplier', 'Score'] + criteria])

        # Download the result as CSV
        st.download_button("Download Ranked Suppliers CSV", df_sorted.to_csv(index=False), "ranked_suppliers.csv")

else:
    st.info("Please upload a CSV to continue. Expected columns: Supplier, Price, Delivery, Quality, Service, Flexibility")
