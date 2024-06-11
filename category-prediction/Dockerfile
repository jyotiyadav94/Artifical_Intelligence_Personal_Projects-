FROM python:3.8

WORKDIR /app

COPY requirements.txt .
COPY setup.py .


RUN apt update -y && apt-get install -y awscli build-essential


RUN pip install --upgrade pip setuptools wheel


RUN pip install -v --no-cache-dir -r requirements.txt


COPY prediction_service ./prediction_service
COPY webapp ./webapp
COPY app.py ./app.py

CMD ["gunicorn", "-b", "0.0.0.0:8080", "--timeout", "300", "app:app"]
