FROM gcr.io/kaggle-gpu-images/python:latest

COPY ./ ./
RUN apt-get update && apt-get upgrade -y \
        && apt-get install -y \
        git

RUN pip install poetry \
        && make develop_no_venv
