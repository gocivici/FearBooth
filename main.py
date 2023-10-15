import cv2
from deepface import DeepFace

# img = cv2.imread("test.jpg")
cam = cv2.VideoCapture(1)
if cam.isOpened():
    while True:
        ret, img = cam.read()
        if ret:
            predictions = DeepFace.analyze(img,enforce_detection=False,actions=['emotion'])
            print(predictions[0]["emotion"]["fear"])

            cv2.imshow('webcam',img)
        key = cv2.waitKey(5) & 0xFF
        if ord('q') == key:
            break
cam.release()