FROM python:3.9-slim-buster


WORKDIR /app

COPY . .

RUN pip install  gradio
RUN pip install  joblib


ENTRYPOINT python app.py