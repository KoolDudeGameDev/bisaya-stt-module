from django.urls import path
from .views import SpeechToTextAPIView

urlpatterns = [
    path('transcribe/', SpeechToTextAPIView.as_view(), name='speech_to_text'),
]
# This file defines the URL patterns for the speech_to_text app.
# It includes a single endpoint for transcribing audio files.
