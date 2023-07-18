From python:3.9-slim-buster

# set working directory to /app
WORKDIR /app                                

#source path to destination path ie. current directory to /app
COPY . .     

#install dependencies
RUN pip install -r requirement.txt         
RUN python 3.Mlflow_Autolog.py

# cmd to launch app when container is run
CMD ["mlflow", "ui"]

