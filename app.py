import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# --- 1. Page Config (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Admission Predictor", page_icon="🎓")

# --- 2. Smart Model Loading ---
# This function tries to load the saved model.
# If it fails (file missing or version mismatch), it trains a new one instantly.
@st.cache_resource
def get_model():
    # Try loading the saved model first
    if os.path.exists('model.pkl'):
        try:
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            print(f"Error loading model.pkl: {e}. Retraining...")
    
    # If loading failed, train the model right now
    try:
        if not os.path.exists('admission_data.csv'):
            st.error("Error: 'admission_data.csv' not found. Please upload it to your GitHub repo.")
            st.stop()
            
        df = pd.read_csv('admission_data.csv')
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Drop Serial No. if it exists
        if 'Serial No.' in df.columns:
            df = df.drop(columns=['Serial No.'])
            
        X = df.drop(columns=['Chance of Admit'])
        y = df['Chance of Admit']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        return model
    except Exception as e:
        st.error(f"Critical Error: Could not train model. Details: {e}")
        st.stop()

# Load the model using our smart function
model = get_model()

# --- 3. UI Design ---
st.title("🎓 Master's Admission Predictor")
st.write("Enter your credentials below to calculate your admission chances.")

st.markdown("---") # A divider line

# --- 4. Input Form ---
col1, col2 = st.columns(2)

with col1:
    gre = st.number_input("GRE Score (0-340)", min_value=0, max_value=340, value=310)
    toefl = st.number_input("TOEFL Score (0-120)", min_value=0, max_value=120, value=105)
    rating = st.slider("University Rating (1-5)", 1, 5, 3)
    sop = st.slider("Statement of Purpose (1-5)", 1.0, 5.0, 3.5, step=0.5)

with col2:
    lor = st.slider("Letter of Recommendation (1-5)", 1.0, 5.0, 3.5, step=0.5)
    cgpa = st.number_input("CGPA (0-10)", min_value=0.0, max_value=10.0, value=8.5, step=0.01)
    research = st.radio("Do you have Research Experience?", ["No", "Yes"])

# Convert Research "Yes/No" to 1/0
research_val = 1 if research == "Yes" else 0

st.markdown("---")

# --- 5. Prediction Logic ---
if st.button("Predict My Chance 🚀"):
    # Create the input array matching the training data order
    user_input = np.array([[gre, toefl, rating, sop, lor, cgpa, research_val]])
    
    # Predict
    prediction = model.predict(user_input)
    chance = prediction[0]
    
    # Display Result
    st.subheader(f"Your Chance of Admission is: **{chance * 100:.1f}%**")
    
    # Custom Message based on result
    if chance > 0.85:
        st.success("🎉 Excellent! You have a very high chance.")
    elif chance > 0.65:
        st.info("👍 Good chance. A strong SOP could help boost this.")
    else:
        st.warning("⚠️ It might be tough. Consider applying to safe universities.")