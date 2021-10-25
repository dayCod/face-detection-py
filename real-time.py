import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithms)
trained_face_data = cv2.CascadeClassifier("C:/LEARN PYTHON/ai_learn/face_recognition/haarcascade_frontalface_default.xml")

# Capture video from webcam
webcam = cv2.VideoCapture(0)
# Note 0 => for default webcam
# you can also put a video inside videocapture braces

while True:
    # Read the current Frame
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    face_coordinate = trained_face_data.detectMultiScale(grayscaled)

    # Draw the rectangle
    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (w+x, h+y), (0, 255, 0), 5)
        

    cv2.imshow("dayCod Face Recognition", frame)
    key = cv2.waitKey(1)

    # in ASCII code "q" = 81 and "Q" = 113
    # When i hit q or Q the process will end automatically
    if key==81 or key==113:
        break

# Release the video capture
webcam.release()

    # wait key for rendering a frame if i put 1 it means every 1 miliseconds that will iterating
