from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import cv2
import numpy as np
from django.http import HttpResponse
from rest_framework.parsers import MultiPartParser
from ultralytics import YOLO


class TestView(APIView):
    def get(self, request):
        return Response(
            {"detail": "API Working."},
            status=status.HTTP_200_OK,
        )


model = YOLO("yolo11n.pt")


class ProcessImageView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request):
        uploaded_image = request.FILES.get("image")

        if not uploaded_image:
            return Response(
                {"detail": "No image was provided."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        image_bytes = uploaded_image.read()
        print(image_bytes)

        np_array = np.frombuffer(image_bytes, dtype=np.uint8)

        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return Response(
                {"detail": "Invalid image."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        results = model(image, verbose=False)

        processed_image = results[0].plot()

        success, encoded_image = cv2.imencode(
            ".jpg",
            processed_image,
        )

        if not success:
            return Response(
                {"detail": "Failed to encode processed image."},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return HttpResponse(
            encoded_image.tobytes(),
            content_type="image/jpeg",
        )
