
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt \
    && python -m nltk.downloader stopwords

# Copy the rest of the application
COPY . /app/

#RUN uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Expose the port the app runs on
EXPOSE 8000
EXPOSE 8501

#CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
# Start the Streamlit server
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.enableCORS", "false"]
