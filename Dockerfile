FROM python:3.8-slim-buster

#RUN apt-get update -y
#RUN apt-get install libgomp1

COPY requirements.txt ./requirements.txt

RUN apt-get update && \
	apt-get install -y gzip libgomp1

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt

WORKDIR .

RUN mkdir ./data

COPY src ./src

ENV PATH_TO_DATA_FILE=data/data.gz
ENV PATH_TO_OUTPUT_FILE=data/output.csv
ENV MODEL_DIRECTORY=data/

CMD python src/train.py