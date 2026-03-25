import streamlit as st
import joblib
import pandas as pd
from keras.models import load_model

# set the tab title
st.set_page_config("Review Classification")

# Set the page title
st.title("Positive and Negative Review Classification Project")

# Set header
st.subheader("By Vaishnavi Badade")

# Load the pipeline (data cleaning, preprocessing) and model
pre = joblib.load("text_classification_pre.joblib")
model = load_model("TextClassification.keras")

# Create Input boxes that takes input from the user 

review = st.number_input("review")


# Include a button. After providing all the inputs, user will click on the button. The button should provide the necessary predictions
submit = st.button("Predict Sentiment")

if submit:

    # Apply data cleaning and preprocessing on new data using pre pipeline
    transformed_text = pre.transform(review)

    # predictions
    preds = model.predict(transformed_text)
    if preds>0.5:
        st.subheader("Positive Review")
    else:
        st.subheader("Negative Review")   
