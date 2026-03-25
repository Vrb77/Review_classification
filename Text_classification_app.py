import streamlit as st
import joblib
from keras.models import load_model

# Import custom classes from the separate module — this is required so
# joblib can unpickle the pipeline correctly (pickle stores the module path,
# so the classes must live in a stable, importable module, not __main__)
from preprocessor import TextCleaner, TextTransformer  # noqa: F401

# Set the tab title
st.set_page_config(page_title="Review Classification")

# Set the page title
st.title("Positive and Negative Review Classification Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("text_classification_pre.joblib")
model = load_model("TextClassification.keras")

# Text input (reviews are text, not numbers)
review = st.text_input("Enter your review")

# Predict button
submit = st.button("Predict Sentiment")

if submit:
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        # Pipeline expects a list of strings
        transformed_text = pre.transform([review])

        # Prediction
        preds = model.predict(transformed_text)
        if preds[0][0] > 0.5:
            st.subheader("✅ Positive Review")
        else:
            st.subheader("❌ Negative Review")
