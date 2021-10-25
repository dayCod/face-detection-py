import cv2
from random import randrange

cascade_file = cv2.CascadeClassifier("C:/LEARN PYTHON/ai_learn/face_recognition/haarcascade_frontalface_default.xml")

webcam = cv2.VideoCapture(0)

while True:
    frame_success, frame = webcam.read()

    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_coordinate = cascade_file.detectMultiScale(grayscaled)

    for (x, y, w, h) in frame_coordinate:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 4)

    cv2.imshow("Day Code - Wirandra Face Recoginiton App", frame)
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break

webcam.release()