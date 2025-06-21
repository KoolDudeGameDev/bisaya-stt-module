from django.contrib import admin
from django.urls import path, include  # you MUST include this

urlpatterns = [
    path('admin/', admin.site.urls),

    # Include the URLs from the speech_to_text app
    path('api/stt/', include('speech_to_text.urls')),
]
