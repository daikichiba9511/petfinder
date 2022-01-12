FROM gcr.io/kaggle-gpu-images/python:latest

COPY ./ ./

RUN pip install poetry \
        && make develop_no_venv
