from django.urls import path
from .views import TestView, ProcessImageView

urlpatterns = [
    path("test/", TestView.as_view()),
    path("process-image/", ProcessImageView.as_view()),
]
