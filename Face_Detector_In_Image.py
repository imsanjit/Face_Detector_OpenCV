import cv2
# Load pre-trained algo from opencv
trained_face_file = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Choose image to detect face in
img = cv2.imread('image-1.png')

# Convert image to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = trained_face_file.detectMultiScale(grayscaled_img)
# Draw rectangle around face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# Show image
cv2.imshow('Face Detector', img)
cv2.waitKey()




print('Code compeleted')
