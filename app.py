from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import re
import os
import uvicorn

app = FastAPI(title="Credit Card Fraud Detection API",
    description="""An API that utilises a Machine Learning model that detects a Spam messages""",
    version="1.0.0", debug=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get('/',response_class=HTMLResponse)
def running():
    text='''
    <html>
    <head>
    <link rel="icon" type="image/x-icon" href="static/images/api.png">
    <title>SMS Spam Detection API</title>
    </head>
    <body>
    <div>
    <h1>SMS Spam Detection API</h1>
        <a href="https://github.com/Sibikrish3000/">Github repository</a>
    </div>
    </body>
    </html>
    '''
    return text

class Message(BaseModel):
    message: str

class Feedback(BaseModel):
    message: str
    is_spam: bool

# Load pre-trained models
EXTRA_TREE_MODEL = joblib.load('models/Extra_Tree.pkl')
BERNOULLINB_MODEL = joblib.load('models/BernoulliNB.pkl')
FEEDBACK_CSV = 'feedback.csv'


def preprocess_message(message):
    message = re.sub(r'\W', ' ', message)
    tokens = word_tokenize(message.lower())
    stemmed_words = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(stemmed_words)

@app.post('/predict')
async def predict(message: Message, model: str = Query(...)):

    if model == 'ExtraTree':
        prediction = EXTRA_TREE_MODEL.predict([message.message])[0]
    elif model == 'NaiveBayes':
        prediction = BERNOULLINB_MODEL.predict([message.message])[0]
    else:
        return {"error": "Invalid model selection"}

    return {"prediction": int(prediction)}

feedback_data = []

@app.post('/feedback')
async def feedback(feedback:Feedback):
    processed_message = feedback.message
    label = 1 if feedback.is_spam else 0
    feedback_data.append((processed_message, label))
    df = pd.DataFrame(feedback_data, columns=['message', 'label'])
    if not os.path.exists(FEEDBACK_CSV):
        df.to_csv(FEEDBACK_CSV, index=False)
    else:
        df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    feedback_data.clear()
    return {'message': 'Feedback Received'}

#if __name__ == '__main__':
    #uvicorn.run(app, host='127.0.0.1', port=8000)
