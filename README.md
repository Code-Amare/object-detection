# YOLO Object Detection API

A small Django REST Framework project for learning how **YOLO object detection** works from end to end.

The API accepts an uploaded image, runs it through a pretrained YOLO model, draws the detected objects on the image, and returns the processed image as a JPEG.

> This is primarily a learning project. The goal is to understand the complete image-to-detection pipeline before building more advanced computer-vision applications.

## What I'm Learning

This project demonstrates the basic object-detection pipeline:

1. Receive an image through an HTTP API.
2. Read the uploaded file as bytes.
3. Convert the bytes into an image that OpenCV can process.
4. Run the image through a pretrained YOLO model.
5. Get the detection results.
6. Draw bounding boxes, class labels, and confidence scores.
7. Encode the processed image as JPEG.
8. Return the processed image to the client.

The overall flow is:

```text
Client
  ↓
HTTP Multipart Upload
  ↓
Django REST Framework
  ↓
Uploaded Image Bytes
  ↓
NumPy
  ↓
OpenCV
  ↓
YOLO
  ↓
Detection Results
  ↓
Annotated Image
  ↓
JPEG Encoding
  ↓
HTTP Response
  ↓
Client
```

## Tech Stack

- **Python**
- **Django**
- **Django REST Framework**
- **OpenCV (`cv2`)**
- **NumPy**
- **Ultralytics YOLO**
- **YOLO11 Nano (`yolo11n.pt`)**

## Project Structure

A simplified structure looks like this:

```text
object-detection/
├── backend/
│   ├── api/
│   ├── core/
│   ├── manage.py
│   ├── requirements.txt
│   ├── yolo11n.pt
│   └── venv/
└── README.md
```

The `backend` directory contains the Django project and API.

The `yolo11n.pt` file is the pretrained YOLO model used by the API.

## How It Works

### 1. Loading the YOLO Model

The model is loaded when the Django application starts:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
```

Instead of training a model from scratch, this project starts with a pretrained model so that the focus can be on understanding inference and image processing.

---

### 2. Receiving an Image

The API expects the client to send an image using:

```text
multipart/form-data
```

The uploaded file is accessed with:

```python
uploaded_image = request.FILES.get("image")
```

The field name must therefore be:

```text
image
```

If no image is supplied, the API returns:

```json
{
  "detail": "No image was provided."
}
```

with HTTP status `400 Bad Request`.

---

### 3. Reading the Uploaded File

The uploaded image is first read as raw bytes:

```python
image_bytes = uploaded_image.read()
```

At this point, it is not yet an OpenCV image. It is simply binary data.

---

### 4. Converting Bytes to a NumPy Array

NumPy converts the raw bytes into an array of unsigned 8-bit integers:

```python
np_array = np.frombuffer(
    image_bytes,
    dtype=np.uint8,
)
```

This gives OpenCV a format it can decode.

---

### 5. Decoding the Image with OpenCV

OpenCV converts the NumPy array into an image:

```python
image = cv2.imdecode(
    np_array,
    cv2.IMREAD_COLOR,
)
```

If OpenCV cannot decode the file, the API returns:

```json
{
  "detail": "Invalid image."
}
```

with HTTP status `400 Bad Request`.

This is an important part of the pipeline because the API does not save the uploaded image to disk before processing it.

---

### 6. Running Object Detection

The image is passed to YOLO:

```python
results = model(
    image,
    verbose=False,
)
```

YOLO analyzes the image and returns detection results.

For object detection, the important information includes things such as:

- **Bounding boxes** — where an object is located.
- **Class** — what the detected object is.
- **Confidence** — how confident the model is about the detection.

Conceptually, a detection can be thought of as:

```text
Object:
    class = "person"
    confidence = 0.91

Bounding box:
    x1, y1
    x2, y2
```

---

### 7. Drawing the Results

The first result is rendered directly onto the image:

```python
processed_image = results[0].plot()
```

The resulting image contains visual annotations such as:

```text
+--------------------------------+
|                                |
|       ┌──────────────┐         |
|       │    person    │         |
|       │    0.91      │         |
|       └──────────────┘         |
|                                |
+--------------------------------+
```

This makes the model's predictions easy to see.

---

### 8. Encoding the Result

The processed image is encoded as JPEG:

```python
success, encoded_image = cv2.imencode(
    ".jpg",
    processed_image,
)
```

If encoding fails, the API returns HTTP `500 Internal Server Error`.

---

### 9. Returning the Image

Finally, Django sends the encoded image back to the client:

```python
return HttpResponse(
    encoded_image.tobytes(),
    content_type="image/jpeg",
)
```

So the response is an actual image rather than JSON.

The client can display the returned JPEG directly.

## API Endpoints

### Test Endpoint

A simple endpoint is available to verify that the API is running.

Response:

```json
{
  "detail": "API Working."
}
```

Expected status:

```text
200 OK
```

### Process Image Endpoint

The image-processing endpoint accepts:

```text
POST
Content-Type: multipart/form-data
```

with the image attached under:

```text
image
```

Successful requests return:

```text
Content-Type: image/jpeg
```

The response is the processed image containing the YOLO detections.

## Example Request

Using `curl`:

```bash
curl -X POST   -F "image=@example.jpg"   http://127.0.0.1:8000/<your-image-endpoint>   --output result.jpg
```

The returned file can then be opened as:

```text
result.jpg
```

Replace `<your-image-endpoint>` with the URL configured in your Django `urls.py`.

## Understanding the Important Libraries

### Django REST Framework

DRF handles the HTTP API layer.

In this project it is responsible for receiving the request and creating the API response.

### NumPy

NumPy represents the raw image bytes as an array.

```python
np.frombuffer(...)
```

This lets OpenCV work with the uploaded image without first writing it to disk.

### OpenCV

OpenCV handles image operations.

In this project it is used to:

- Decode the uploaded image.
- Encode the processed image as JPEG.

Later, OpenCV can also be used for resizing, cropping, color conversion, video processing, webcam input, and many other computer-vision operations.

### YOLO

YOLO is the machine-learning part of the pipeline.

The model looks at an image and predicts objects that it has learned to recognize.

The key idea behind object detection is that the model does not only answer:

```text
"What is in this image?"
```

It also answers:

```text
"Where is it?"
```

That is why object detection produces bounding boxes.

## Object Detection vs Classification

This distinction is important while learning computer vision.

### Image Classification

Classification answers:

```text
What is in the image?
```

Example:

```text
Image → Dog
```

### Object Detection

Detection answers:

```text
What objects are present, and where are they?
```

Example:

```text
Image
  ├── Dog → bounding box
  ├── Person → bounding box
  └── Car → bounding box
```

A single image can contain multiple detected objects.

## Why Start with a Pretrained Model?

Training a detection model requires:

- A dataset
- Labeled images
- Bounding-box annotations
- Training configuration
- Significant compute
- Evaluation and experimentation

Using a pretrained model allows this project to focus first on understanding **inference**:

```text
Input Image → Model → Predictions
```

After understanding inference, the next step can be learning how to train or fine-tune YOLO on a custom dataset.

## Current Implementation

The current implementation intentionally keeps the pipeline simple:

```python
uploaded_image
      ↓
image_bytes
      ↓
NumPy array
      ↓
OpenCV image
      ↓
YOLO inference
      ↓
results[0].plot()
      ↓
JPEG
      ↓
HTTP response
```

There is no database requirement for the image-processing operation itself, and the uploaded image is processed in memory.

## Current Limitations

This project is intentionally basic. Some things that can be improved later include:

- Returning detection data as JSON in addition to the processed image.
- Supporting configurable confidence thresholds.
- Handling multiple images in one request.
- Adding file-size and file-type validation.
- Resizing very large images before inference.
- Adding GPU/CPU configuration.
- Returning processing time and model information.
- Processing video files.
- Processing webcam streams.
- Saving detection results.
- Adding authentication and permissions.
- Separating model loading and inference into a cleaner service layer.
- Containerizing the application for deployment.

## Learning Roadmap

A good progression for this project is:

```text
1. Understand image bytes
        ↓
2. Learn NumPy arrays
        ↓
3. Learn OpenCV image operations
        ↓
4. Understand YOLO inference
        ↓
5. Understand bounding boxes
        ↓
6. Understand confidence scores
        ↓
7. Return detection results as JSON
        ↓
8. Process multiple objects
        ↓
9. Process video
        ↓
10. Process webcam input
        ↓
11. Learn datasets and annotations
        ↓
12. Train/fine-tune YOLO
        ↓
13. Build a production-ready computer-vision API
```

## Goal

The long-term goal of this project is not just to make an API that detects objects.

It is to understand **why every step exists**:

```text
HTTP
 ↓
File Upload
 ↓
Binary Data
 ↓
NumPy
 ↓
OpenCV
 ↓
YOLO
 ↓
Predictions
 ↓
Bounding Boxes
 ↓
Image Rendering
 ↓
Encoding
 ↓
HTTP Response
```

Understanding this pipeline provides a foundation for building more advanced applications such as:

- Real-time object detection
- Video analytics
- Camera-based applications
- People counting
- Custom object detection
- Image moderation
- Smart surveillance systems
- Computer-vision features inside web and mobile applications

## License

This repository is intended as a personal learning project.
