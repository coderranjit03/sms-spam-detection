import time
import requests
import threading
import uvicorn
from app import app
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
from PIL import Image
#import nltk
import streamlit as st

icon =Image.open("static/images/icon.png")
about = open("about.md")
st.set_page_config(
        page_title="SMS SPAM DETECTION",
        page_icon=icon,
        layout='wide'
    )
st.markdown(
        f"""
            <style>
            .stApp {{
                background-image: url('https://github.com/Sibikrish3000/sms-spam-detection/blob/main/static/images/cover.jpg?raw=true');
                background-attachment: fixed;
                background-repeat: no-repeat;
                background-size: cover;
            }}
             .st-emotion-cache-1avcm0n{{
            background-image: url('https://github.com/Sibikrish3000/sms-spam-detection/blob/main/static/images/cover.jpg?raw=true');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            }}
            </style>
            """,
        unsafe_allow_html=True
    )


def close_port(port):
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port:
            print(f"Closing port {port} by terminating PID {conn.pid}")
            process = psutil.Process(conn.pid)
            process.terminate()
def run_fastapi():
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f'Error running fastapi:{e}')
        close_port(8000)

fastapi_thread = threading.Thread(target=run_fastapi)
fastapi_thread.daemon = True
fastapi_thread.start()
time.sleep(2)
# Check if NLTK data is downloaded, download if not



stemmer = PorterStemmer()
# Attempt to load stopwords with error handling
try:
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"An error occurred while loading NLTK stopwords: {e}")
    stop_words = set()

def preprocess_message(message):
    message = re.sub(r'\W', ' ', message)
    tokens = word_tokenize(message.lower())
    stemmed_words = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(stemmed_words)

# Main Streamlit app
def main():
    #st-emotion-cache-1avcm0n
    st.title('SMS Spam Detection Webapp')
    st.image('static/images/spam.png',width=720)
    st.subheader('SMS Spam Detection Webapp Using FastAPI')


    message = st.text_area('Enter your SMS message here:')
    model = st.selectbox('Select Model:', ("ExtraTree", "NaiveBayes"))

    processed_message = preprocess_message(message)
    payload = {"message": processed_message}

    if st.button('Predict'):

        if message:
            response = requests.post(f'http://127.0.0.1:8000/predict?model={model}', json=payload)
            if response.status_code == 200:
                prediction = response.json().get("prediction", "Error")
                if prediction == 1:
                    st.error("The message is classified as **spam**.")
                else:
                    st.success("The message is classified as **not spam**.")
            else:
                st.error("Error in prediction. Please try again.")
        else:
            st.error("Please enter a message")

    st.write("Feedback")
    is_spam = st.checkbox("Is it Spam", value=False)
    if st.button("Submit Feedback"):
        if message:
            feedback_payload = {
                "message": processed_message,
                "is_spam": is_spam
            }
            feedback_response = requests.post("http://127.0.0.1:8000/feedback", json=feedback_payload)

            if feedback_response.status_code == 200:
                st.success("Thank you for your feedback!")
            else:
                st.error("Error in submitting feedback. Please try again.")
        else:
            st.error("Please enter a feedback message.")

    with st.expander("About"):
        st.title("SMS Spam Detection Webapp")
        st.markdown(about.read(),unsafe_allow_html=True)
    st.warning("Please press buttons after enter the messages")

    st.markdown('---')
    st.markdown('@Sibi krishnamoorthy')
if __name__ == "__main__":
    main()
    fastapi_thread.join()




