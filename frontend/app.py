import streamlit as st
import requests

st.set_page_config(page_title="Next Word Predictor", layout="centered")
st.title("🔮 Next Word Prediction using NLP")

text = st.text_input("Enter text:")
n_words = st.slider("Number of words to predict:", 1, 5, 1)

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"text": text, "n_words": n_words}
        )
        result = response.json()
        st.success(result["prediction"])
