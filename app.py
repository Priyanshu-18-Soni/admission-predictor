import streamlit as st
import pickle
import numpy as np

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found! Please run train_model.py first.")
    st.stop()

# --- Page Config ---
st.set_page_config(page_title="Admission Predictor", page_icon="🎓")

# --- UI Design ---
st.title("🎓 Master's Admission Predictor")
st.write("Enter your credentials below to calculate your admission chances.")

st.markdown("---") # A divider line

# --- Input Form ---
# We use columns to make it look professional
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

# --- Prediction Logic ---
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