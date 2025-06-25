import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .utils import transcribe_with_wav2vec2, is_transcription_confident
from .whisper_model import transcribe_with_whisper
from .serializers import AudioFileSerializer
import mimetypes


class SpeechToTextAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]
    serializer_class = AudioFileSerializer

    def post(self, request, *args, **kwargs):
        serializer = AudioFileSerializer(data=request.data)
        if serializer.is_valid():
            audio_file = serializer.validated_data['file']

            # Check file size limit
            if audio_file.size > 10 * 1024 * 1024:  # Limit to 10 MB
                return Response({"error": "File size exceeds 10 MB limit."}, status=status.HTTP_400_BAD_REQUEST)
            print(f"Received file: {audio_file.name}, size: {audio_file.size} bytes")  # Debugging line

            # Check if the file is an audio file
            content_type, _ = mimetypes.guess_type(audio_file.name)
            if not content_type or not content_type.startswith('audio'):
                return Response({"error": "Invalid file type. Please upload an audio file."}, status=status.HTTP_400_BAD_REQUEST)

            temp_path = f"temp_{audio_file.name}"
            with open(temp_path, 'wb+') as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
            print(f"Temporary file created at: {temp_path}")  # Debugging line

            try:
                primary_text = transcribe_with_wav2vec2(temp_path)
                if is_transcription_confident(primary_text):
                    final_text = primary_text
                    engine_used = "Wav2Vec2"
                else:
                    final_text = transcribe_with_whisper(temp_path)
                    engine_used = "Whisper-fallback"

            except Exception as e:
                print(f"Error during transcription: {e}")  # Debugging line
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                if os.path.exists(temp_path):
                    # Debugging line
                    print(f"Removing temporary file: {temp_path}")
                    os.remove(temp_path)

            return Response({
                "transcription": final_text,
                "engine": engine_used
            }, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
