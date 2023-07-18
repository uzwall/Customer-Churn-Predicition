FROM python:3.9-slim-buster

# Set working directory to /app
WORKDIR /app

# Copy source code to the container's working directory
COPY . .

# Install dependencies
RUN pip install -r requirements.txt
RUN python 3.Mlflow_Autolog.py

# Command to run mlflow ui and python script
# CMD ["mlflow","ui"]
entrypoint mlflow ui --host='0.0.0.0' --port='5000'