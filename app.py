33333333333333333import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

# Streamlit Page Config
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    page_icon="📰"
)

st.title("📰 Fake News Detector")
st.markdown("### Detect whether a news article is *Fake* or *Real*.") 
st.write("Paste your news content below and click **Detect Fake News** to analyze it.")

# Load Model & Tokenizer
@st.cache_resource
def load_model():
    model_dir = "bert_fakenews_model"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

# Prediction Function
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    return np.argmax(probs), probs

# User Input
user_input = st.text_area("📝 Enter the news article text below:", height=200, placeholder="Type or paste news article content here...")

if st.button("🔍 Detect Fake News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        label, probs = predict(user_input)
        st.write(f"**Prediction Probabilities**\n\n- Fake → `{probs[0]:.2f}`\n- Real → `{probs[1]:.2f}`")
        if probs[0] > 0.6:
            st.error(f"🚫 This news is **Fake** (Confidence: {probs[0]:.2f})")
        elif probs[1] > 0.6:
            st.success(f"✅ This news is **Real** (Confidence: {probs[1]:.2f})")
        else:
            st.warning("🤔 Model is unsure. Please verify manually.")

st.markdown("---")
st.caption("Made with ❤️ using BERT and Streamlit")
