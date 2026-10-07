from ultralytics import YOLO
import cv2
import os
import numpy as np

face_model = YOLO("yolov11m-face.pt").to("cuda")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

folder_path = "train-pics/"

folders = {1: "Amare", 2: "Temesgen", 3: "Alazar", 4: "Misgana"}

proccessed_image = []
faces = []
lables = []


for lable, folder in folders.items():
    files = os.listdir(folder_path + folder)
    for file in files:
        image = cv2.imread(folder_path + folder + "/" + file)
        face_result = face_model(image, verbose=False)
        if len(face_result[0].boxes) == 0:
            continue

        proccessed_image.append(face_result)
        left, top, right, bottom = face_result[0].boxes.xyxy[0].int()
        face = image[top:bottom, left:right]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (200, 200))
        lables.append(lable)
        faces.append(face)

face_recognizer.train(faces, np.array(lables))
face_recognizer.write("face_recognizer.yml")

cv2.imshow("Result", faces[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
