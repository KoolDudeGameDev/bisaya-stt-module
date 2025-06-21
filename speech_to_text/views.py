import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from .utils import transcribe_audio_file
from .serializers import AudioFileSerializer


class SpeechToTextAPIView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = AudioFileSerializer(data=request.data)
        if serializer.is_valid():
            audio_file = serializer.validated_data['file']

            temp_path = f"temp_{audio_file.name}"
            with open(temp_path, 'wb+') as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)

            try:
                transcription = transcribe_audio_file(temp_path)
            except Exception as e:
                return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            finally:
                os.remove(temp_path)

            return Response({"transcription": transcription}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
