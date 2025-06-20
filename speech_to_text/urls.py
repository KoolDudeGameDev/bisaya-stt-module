from django.urls import path
from .views import TranscribeAPI

urlpatterns = [
    path('api/transcribe/', TranscribeAPI.as_view(), name='transcribe'),
    # Add other API endpoints here as needed
]
