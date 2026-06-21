import streamlit as st
import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

st.set_page_config(page_title="Sentiment Analysis", page_icon="🤖")

st.title("🤖 AI Sentiment Analysis")
st.write("Analyze whether a sentence is Positive, Negative or Neutral")

text = st.text_area("Enter Text")

if st.button("Analyze"):
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)

    if prediction[0] == "positive":
        st.success("😊 Positive Sentiment")

    elif prediction[0] == "negative":
        st.error("😠 Negative Sentiment")

    else:
        st.info("😐 Neutral Sentiment")