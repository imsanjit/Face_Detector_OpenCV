import cv2

# Load pre-trained algo from opencv
trained_face_file = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Cature video from webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read current frame
    read_successful, frame = webcam.read()
    # safe coding
    if read_successful:
        # Convert image to grayscale
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # Detect faces
    faces = trained_face_file.detectMultiScale(grayscaled_img)
    # Draw rectangle around face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


    # Show image
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    # stop if q key is pressed
    if key == 81 or key == 113:
        break

# Release webcam frame
webcam.release()