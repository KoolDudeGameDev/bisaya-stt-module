from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
from django.core.files.uploadedfile import SimpleUploadedFile

# Create your tests here.


class SpeechToTextAPITestCase(APITestCase):
    def test_api_transcription_endpoint(self):
        # Minimal valid WAV header
        audio_content = b'\x52\x49\x46\x46\x24\x08\x00\x00\x57\x41\x56\x45'
        audio_file = SimpleUploadedFile(
            "test.wav", audio_content, content_type="audio/wav")

        url = reverse('speech_to_text')
        response = self.client.post(
            url, {'file': audio_file}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('transcription', response.data)
