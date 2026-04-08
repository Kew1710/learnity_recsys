FROM python:3.12-slim

WORKDIR /app

COPY requirements-shared.txt .
RUN pip install --no-cache-dir -r requirements-shared.txt

COPY . .

ENV PYTHONPATH=/app
