import streamlit as st
import pickle
import os
import sys

sys.path.append("src")
from utils import clean_text

# Load model
model = pickle.load(open("model/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.set_page_config(page_title="AI Spam Detector", layout="centered")

st.title("📧 AI Email Spam Detection System")
st.write("Enter an email message to check whether it is Spam or Not Spam.")

user_input = st.text_area("Email Message")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])
        prediction = model.predict(transformed)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM!")
        else:
            st.success("✅ This is NOT SPAM!")
