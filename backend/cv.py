import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt").to("cuda")
face_model = YOLO("yolov11m-face.pt").to("cuda")
photo = cv2.imread("image.png")


cap = cv2.VideoCapture(0)

while cv2.waitKey(1) != ord("x"):
    _, frame = cap.read()
    objs = model(frame, verbose=False)
    final_frame = face_model(objs[0].plot(), verbose=False)

    cv2.imshow("Image", final_frame[0].plot())
    # cv2.imshow("Image", frame)

cv2.waitKey(5000)
cap.release()
cv2.destroyAllWindows()
