import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
import numpy as np

# Page Config
st.set_page_config(
    page_title="Fake News Detector",
    layout="centered",
    page_icon="📰"
)

# 🎨 Custom CSS: Background + Dark Overlay + Typing Effect
def add_custom_style():
    st.markdown(
        """
        <style>
        /* Full-screen background */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                        url("https://th.bing.com/th/id/OIP.7ZNmjomVugIIAYs1n2_cKwHaEK?w=326&h=183&c=7&r=0&o=5&dpr=1.3&pid=1.7");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* White container with transparency */
        .styled-box {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            margin-bottom: 20px;
        }

        /* Typing animation */
        .typing-header {
            font-size: 24px;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            border-right: 3px solid #1f77b4;
            width: 0;
            animation: typing 3s steps(40, end) forwards, blink 0.75s step-end infinite;
        }

        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }

        @keyframes blink {
            50% { border-color: transparent }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_custom_style()

# 🔤 Header with Typing Effect
#st.markdown('<div class="styled-box">', unsafe_allow_html=True)
st.markdown('<div class="typing-header">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown("### Detect whether a news article is *Fake* or *Real*.")
st.write("Paste your news content below and click **Detect Fake News** to analyze it.")
#st.markdown('</div>', unsafe_allow_html=True)

# 🧠 Load BERT Model & Tokenizer
@st.cache_resource
def load_model():
    model_dir = "bert_fakenews_model"
    model = TFAutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return model, tokenizer

model, tokenizer = load_model()

# 🔍 Prediction Function
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    return np.argmax(probs), probs

# 📝 User Input
#st.markdown('<div class="styled-box">', unsafe_allow_html=True)
user_input = st.text_area("📝 Enter the news article text below:", height=200, placeholder="Type or paste news article content here...")
st.markdown('</div>', unsafe_allow_html=True)

# 🎯 Detect Button + Result
if st.button("🔍 Detect Fake News"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        label, probs = predict(user_input)
        #st.markdown('<div class="styled-box">', unsafe_allow_html=True)
        st.write(f"**Prediction Probabilities**\n\n- Fake → `{probs[0]:.2f}`\n- Real → `{probs[1]:.2f}`")
        if probs[0] > 0.6:
            st.error(f"🚫 This news is **Fake** (Confidence: {probs[0]:.2f})")
        elif probs[1] > 0.6:
            st.success(f"✅ This news is **Real** (Confidence: {probs[1]:.2f})")
        else:
            st.warning("🤔 Model is unsure. Please verify manually.")
        #st.markdown('</div>', unsafe_allow_html=True)

# 📌 Footer
st.markdown("---")
st.caption("Made with ❤️ using BERT and Streamlit")
