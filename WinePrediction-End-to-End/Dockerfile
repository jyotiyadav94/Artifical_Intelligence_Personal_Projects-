FROM python:3.7

WORKDIR /app

COPY requirements.txt .
COPY setup.py .

RUN pip install -v --no-cache-dir -r requirements.txt

COPY prediction_service ./prediction_service
COPY webapp ./webapp
COPY app.py ./app.py

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]