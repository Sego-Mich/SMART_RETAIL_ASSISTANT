import streamlit as st
import pickle
import numpy as np
import os
from keras.models import load_model

# --- Function to load artifacts ---
@st.cache_resource(show_spinner=False)
def load_artifacts(model_path, label_path, vectorizer_path):
    try:
        model = load_model(model_path)
        with open(label_path, "rb") as f:
            label_encoder = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        return model, label_encoder, vectorizer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

# --- UI ---
st.title("Smart Retail Assistant – Inference")
st.subheader("Load your model and make predictions")

# Model input option
model_source = st.radio("Select model loading method:", ("Use default path", "Upload .keras file"))

if model_source == "Use default path":
    model_path = "checkpoints/best_model_overall.keras"
else:
    uploaded_model = st.file_uploader("Upload your .keras model", type=["keras"])
    if uploaded_model is not None:
        model_path = uploaded_model

# Paths to encoder and vectorizer (assumed pre-saved)
label_path = "label_encoder.pkl"
vectorizer_path = "vectorizer.pkl"

# Load model only when ready
if st.button("Load Model"):
    model, label_encoder, vectorizer = load_artifacts(model_path, label_path, vectorizer_path)

    if model:
        st.success("✅ Model and encoders loaded successfully!")

        # Prediction UI
        user_input = st.text_area("Enter product description to predict category:")

        if st.button("Predict"):
            if user_input.strip() == "":
                st.warning("Please enter a valid description.")
            else:
                X_input = vectorizer.transform([user_input])
                prediction = model.predict(X_input.toarray())
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

                st.success(f"Predicted Category: **{predicted_label}**")
else:
    st.info("Please load a model first to proceed.")
