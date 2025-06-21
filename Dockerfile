# Use official Python image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libsndfile1

# Copy Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your Django project into the container
COPY . .

# Run Django development server (replace with gunicorn in production)
CMD ["python", "stt_service/manage.py", "runserver", "0.0.0.0:8000"]
