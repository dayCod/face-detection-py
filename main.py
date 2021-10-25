import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithms)
trained_face_data = cv2.CascadeClassifier("C:/LEARN PYTHON/ai_learn/face_recognition/haarcascade_frontalface_default.xml")

# Choose an image to detect faces in
img = cv2.imread("C:/LEARN PYTHON/ai_learn/face_recognition/famm.jpeg")


# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (w+x, h+y), (randrange(256),randrange(256),randrange(256)), 5)

# second tuples is put the first two pack of number in the face coordinates
# Third tuples is put the second two pack of number + first two pack of num in the face coordinates
# Set the color -> in open cv start from BGR which means Blue green red if first column in tuples is 255 it can be blue color
# and the last item in rectangle is thickness of the rectangle

# print(face_coordinates)

# Display th image with the faces 
cv2.imshow("dayCode Face Recoginiton", img)
cv2.waitKey()

print("Code Completed")