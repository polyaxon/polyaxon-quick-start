FROM python:3.13

# This dockerfile emulate the behavior of Polyaxon v0 using similar tricks for leverage the build cache
LABEL maintainer="Polyaxon authors <contact@polyaxon.com>"

WORKDIR /code

COPY requirements.txt /code

RUN pip install --no-cache-dir -r /code/requirements.txt

COPY scheduling/model.py /code/scheduling/model.py
