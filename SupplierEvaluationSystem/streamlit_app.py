import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv("SupplierEvaluationSystem/updated_dummy_supplier_data.csv")

# Normalize the data
criteria = ['Price', 'Delivery', 'Quality', 'Service', 'Flexibility']
scaler = MinMaxScaler()
df[criteria] = scaler.fit_transform(df[criteria])

# Feature columns
X = df[criteria]  # Independent variables
y = df['Score']  # Dependent variable (Score)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predict on test data
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Training Complete: Mean Squared Error: {mse:.4f}")

# Streamlit UI
st.title("🔍 AI-Driven Supplier Evaluation Portal")

# Sidebar for user input
price_weight = st.slider('💰 Price', -1.0, 1.0, -0.3)
delivery_weight = st.slider('🚚 Delivery', -1.0, 1.0, -0.15)
quality_weight = st.slider('📦 Quality', -1.0, 1.0, 0.25)
service_weight = st.slider('🔧 Service', -1.0, 1.0, 0.2)
flexibility_weight = st.slider('🤹 Flexibility', -1.0, 1.0, 0.1)

# Predict supplier score based on user input
weights = [price_weight, delivery_weight, quality_weight, service_weight, flexibility_weight]
user_input = pd.DataFrame([weights], columns=criteria)

# Normalize user input data
user_input[criteria] = scaler.transform(user_input[criteria])

# Predict score for user input
predicted_score = rf_model.predict(user_input)

# Display the prediction
st.write(f"Predicted Supplier Score: {predicted_score[0]:.2f}")
