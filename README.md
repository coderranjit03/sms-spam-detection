<p align="center">
<a href = "https://github.com/Sibikrish3000/sms-spam-detection" > <img src = "https://github.com/Sibikrish3000/sms-spam-detection/blob/main/static/images/spam.png?raw=true" alt = "sms spam image"  width=500 height=280> </a>
</p>
<h1 align="center"> SMS Spam Detection Web Application </h1>

<p align="center">
This application leverages machine learning to detect spam messages
</p>

<p align="center">
<a href="https://github.com/Sibikrish3000/sms-spam-detection/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Sibikrish3000/sms-spam-detection" alt="GitHub license"></a>
<a href="https://github.com/Sibikrish3000/sms-spam-detection/stargazers"><img src="https://img.shields.io/github/stars/Sibikrish3000/sms-spam-detection?style=social" alt="GitHub stars"></a>
<a href="https://github.com/Sibikrish3000/sms-spam-detection/issues"><img src="https://img.shields.io/github/issues/Sibikrish3000/sms-spam-detection" alt="GitHub issues">
</p>
<p align="center">
<a href="https://scikit-learn.org/"><img src=https://img.shields.io/badge/sklearn-darkorange.svg?style=flat&logo=scikit-learn&logoColor=white alt="sklearn"></a>
<a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-yellow.svg?style=flat&logo=python&logoColor=white" alt="language"></a>
<a href="https://fastapi.tiangolo.com/" ><img src="https://img.shields.io/badge/FastAPI-darkgreen.svg?style=flat&logo=fastapi&logoColor=white " alt="fastapi"></a> <a href="https://hub.docker.com/repository/docker/sibikrish3000/sms-spam-detection/"><img src="https://img.shields.io/badge/Docker-blue?style=flat&logo=docker&logoColor=white" alt= "docker"></a>
<a href="https://www.streamlit.io"><img src="https://img.shields.io/badge/Streamlit-e63946?style=flat&logo=streamlit&logoColor=white" alt="streamlit"></a>
</p>


This repository contains a web application for detecting spam SMS messages. The application uses machine learning models (Extra Trees and Bernoulli Naive Bayes) to classify messages as spam or not spam. The app also allows users to provide feedback on the classification results, which can be used to retrain the models periodically.

[Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
## Try on Streamlit
<p>
<a href="https://www.streamlit.io"><img src="https://img.shields.io/badge/Streamlit-e63946?style=flat&logo=streamlit&logoColor=linear-gradient(360deg, #f093fb 0%, #f5576c 100%)" alt="streamlit" width="160" height="50" ></a>
</p>

## Try on Huggingface Space
<p>
<a href="https://huggingface.co/spaces/sibikrish/sms-spam-detection?theme=dark"><img src="https://img.shields.io/badge/Huggingface-white?style=flat&logo=huggingface&logoSize=amd" alt="huggingface" width="160" height="50" ></a>
</p>



### Features

- **Prediction**: Classify SMS messages as spam or not spam using Extra Trees or Bernoulli Naive Bayes models.
- **Feedback**: Users can provide feedback on the predictions to improve model performance.
- **Continuous Training**: The application supports periodic retraining of models using the feedback data.

## Project Structure

```
/sms-spam-detection
│
├──/model
│   ├── BernoulliNB.pkl
│   └── Extra_Tree.pkl
│
├──/static
│   └──/images
│
├── app.py
├── streamlit_app.py
├── docker_app.py
├── Dockerfile
├── Dockerfile.fastapi
├── docker-compose.yml
├── requirements.txt
````

- `app.py`: Defines the FastAPI application.
- `streamlit_app.py`: Defines the streamlit webapp.
- `docker_app.py`: streamlit webapp for docker
- `Dockerfile`: Dockerfile for building the Docker image.
- `docker-compose.yml`: Docker Compose file for orchestrating the services.
- `requirements.txt`: List of dependencies.
-  `model/`: Directory containing pre-trained machine learning models.
- `static/`: Directory containing static files such as images used in the interface.



### Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Sibikrish3000/sms-spam-detection.git
    cd sms-spam-detection
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download NLTK data**:
    ```
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    ```

## Run Locally

1. **Start the FastAPI Server**:
    ```sh
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
    ```

2. **Run the Streamlit Application**:
    ```sh
    streamlit run streamlit_app.py
    ```
### Using Docker Compose

1. Build and start the containers:
   ```sh
   docker network create AIservice
   ```
    ```sh
    docker-compose up --build
    ```

2. Access the streamlit webapp at [http://localhost:8501](http://localhost:8080).

### Using Docker image

```sh
docker network create AIservice
```
```sh
docker pull sibikrish/sms-spam-detection:latest
docker run sibikrish/sms-spam-detection:latest #or 
docker run -d -p 8501:8501 sibikrish/sms-spam-detection:latest
 ``` 
## Development
### Running in a Gitpod Cloud Environment

**Click the button below to start a new development environment:**

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/Sibikrish3000/sms-spam-detection)

### Usage

- **Enter SMS Message**: Input the SMS message you want to classify.
- **Select Model**: Choose between Extra Trees and Bernoulli Naive Bayes models.
- **Predict**: Click the "Predict" button to see the classification result.
- **Feedback**: Provide feedback on the prediction by marking the message as spam or not spam and submit.

### Continuous Training (CT) in MLOps

Continuous Training (CT) ensures that the machine learning models stay up-to-date with new data and feedback. Here are some suggestions for implementing CT for this application:

#### Online Learning

Online learning is suitable for scenarios where data arrives continuously, and the model needs to update frequently.

- **Implementation**: Implement online learning techniques where models are updated incrementally as new labeled data arrives. 
Use techniques like stochastic gradient descent or mini-batch learning to update models in real-time based on user feedback. Use the `partial_fit()` method available in some scikit-learn models
 (e.g., SGDClassifier,BernoulliNB) to update the model incrementally.
- **Benefits**: The model updates with each new feedback, allowing it to adapt quickly to new patterns.
- **Challenges**: May require more careful tuning and monitoring to ensure model stability.

#### Offline Learning

Offline learning involves retraining the model periodically with the accumulated feedback data.


- **Implementation**: Retrain the model every fixed interval (e.g., daily, weekly) using the feedback data stored in the CSV file.
- **Benefits**: Simpler to implement and manage, as retraining can be scheduled during off-peak times.
- **Challenges**: Model updates less frequently compared to online learning, which may delay the incorporation of new patterns.

#### Partial Fit

Partial fit combines aspects of both online and offline learning.

- **Implementation**: Use models that support the `partial_fit()` method. Collect feedback data over a period and then update the model in smaller batches.
- **Benefits**: Provides a balance between frequent updates and stability.
- **Challenges**: Requires careful management of the batch size and frequency of updates.

### Example Workflow for Offline Learning with Periodic Retraining

1. **Collect Feedback**: Save feedback data into a CSV file.
2. **Scheduled Retraining**: Set up a cron job or similar scheduling tool to retrain the model every 10 days.
3. **Model Update**: Load the feedback data, preprocess it, and retrain the model.
4. **Save Model**: Save the retrained model to a file and replace the old model.

#### Cron Job Example (Linux)

```sh
# Open the crontab editor
crontab -e

# Add the following line to schedule retraining every 10 days
0 0 */10 * * /usr/bin/python3 /path/to/your/retrain_script.py
```

### Retraining Script Example

```python
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier

# Load feedback data
df = pd.read_csv('feedback.csv')

# Preprocess the messages
# Include your preprocessing function here

# Vectorize the messages
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Retrain the model
model = ExtraTreesClassifier()
model.fit(X, y)

# Save the retrained model
joblib.dump(model, 'Extra_Tree.pkl')
```


### License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
