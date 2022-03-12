FROM python:3.8-slim-buster

WORKDIR /srv

COPY requirements.txt .
RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python3.8", "src/hello.py"]
