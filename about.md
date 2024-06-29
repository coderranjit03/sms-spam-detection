## SMS Spam Detection Web Application

This SMS Spam Detection Web Application leverages Machine Learning models served as an API to identify potentially spam SMS messages. The app empowers users to assess message legitimacy based on their content, providing an efficient way to filter out unwanted spam.

### Features:

1. **FastAPI Backend**: The backend of the application is implemented using FastAPI, a modern web framework for building APIs with Python. It exposes an endpoint `/predict` that accepts POST requests with SMS message data and returns predictions. Another endpoint `/feedback` allows users to provide feedback on the predictions.

2. **Streamlit Frontend**: The frontend of the application is implemented using Streamlit, a Python library that allows for the creation of customizable UI components for machine learning models. Users interact with the application through a user-friendly interface where they can input SMS messages and receive predictions.

3. **Models**: The application utilizes ExtraTreeClassifier and Bernoulli Naive Bayes models, leveraging powerful machine learning algorithms for spam detection.

4. **Feedback Mechanism**: Users can provide feedback on the predictions, indicating whether a message was correctly classified as spam or not. This feedback is stored and used to improve the model over time.

### Usage:

- Users can run the application locally by executing the provided Python script.
- They can interact with the application through the Streamlit interface in their web browser, inputting SMS messages and receiving predictions.
- The application provides predictions in real-time, leveraging machine learning models trained on historical SMS data.

### Deployment:

- The application can be deployed locally or on a cloud platform using Docker. Docker containers encapsulate both the FastAPI backend and the Streamlit frontend, making deployment straightforward.
- Additionally, the application can be deployed to a serverless platform like Vercel or Heroku, leveraging their respective deployment methods.

### Future Improvements:

1. Enhance model performance by fine-tuning hyperparameters or using more sophisticated models.
2. Add more features to improve prediction accuracy.
3. Implement user authentication and authorization for secure access to the application.
4. Integrate with a database to store feedback examples for analysis and model improvement.

### Development:

- Developers can extend and enhance the application by adding new features, improving model accuracy, or optimizing performance.
- The codebase is modular and well-structured, facilitating easy maintenance and collaboration among developers.

Overall, this SMS Spam Detection application provides a practical solution for identifying potentially spam messages, helping users keep their inboxes clean and efficient.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/Sibikrish3000/sms-spam-detection/blob/main/LICENSE)  file for details.

The Jupyter notebook, trained model, and accompanying documentation, including Dockerfiles, FastAPI script, and Streamlit Interface script, can be accessed through the GitHub repository linked below:

<p>
<a href="https://github.com/Sibikrish3000/sms-spam-detection"><img src=https://img.shields.io/badge/Github%20Repository-white.svg?style=flat&logo=github&logoColor=black alt="Github repo"></a>
</p>

![size](https://img.shields.io/github/repo-size/Sibikrish3000/sms-spam-detection)

Please feel free to explore and utilize these resources for SMS spam detection purposes.

### [@Sibi krishnamoorthy](https://sibikrish3000.github.io/portfolio/)
___

<h5 align="center">
Sibi krishnamoorthy
</h5><p align="center">
A Data Science enthusiast with a passion for Machine Learning and Artificial Intelligence
</p><p style="color:teal" align="center">
&copy Sibikrish. All rights reserved 2024
</p>

