# Bisaya STT Module Backend API

## Stack:
- Python 3.11
- Django REST Framework
- PyTorch + Transformers (Hugging Face)
- JupyterLab for experiments
- Docker for containerization

## Running Locally:
```
source bisaya-stt-env/bin/activate
python manage.py runserver
```

## API:
`POST /api/transcribe/` (multipart/form-data)
- file: WAV file input
- returns: JSON {"transcript": "text"}
