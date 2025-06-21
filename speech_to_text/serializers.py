from rest_framework import serializers

# Serializers for handling audio file uploads in the API


class AudioFileSerializer(serializers.Serializer):
    file = serializers.FileField()
