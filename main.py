import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

st.title("📩 SMS & Message Spam Detector")
st.write("Enter any text message below to check whether it's legitimate or spam.")

@st.cache_resource
def load_model():
    return joblib.load('spam_pipeline.joblib')

model = load_model()

user_input = st.text_area('Message content: ', placeholder='Type or paste a message here')

if st.button('Process'):
    if user_input == '':
        st.warning('Please enter text to continue')

    else:
        prediction = model.predict([user_input])[0]
        probabilities = model.predict_proba([user_input])[0]

        st.subheader("Result:")
        
        if prediction == 1:
            st.error(f"🚨 **SPAM DETECTED** (Confidence: {probabilities[1]*100:.1f}%)")
        else:
            st.success(f"✅ **LEGITIMATE (HAM)** (Confidence: {probabilities[0]*100:.1f}%)")