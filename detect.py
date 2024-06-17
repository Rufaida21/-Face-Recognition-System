import cv2
import numpy as np
import os
import sqlite3

# Initialize the face detector and camera
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if the training data file exists
training_data_path = "Rekognizer/trainingdata.yml"
if not os.path.exists(training_data_path):
    print(f"Error: Training data file not found at {training_data_path}.")
    exit()

# Read the training data
recognizer.read(training_data_path)


# Function to get profile information from the database
def getProfile(id):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.execute("SELECT * FROM STUDENTS WHERE id=?", (id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile


try:
    while True:
        # Read a frame from the camera
        ret, img = cam.read()

        # Check if the frame is read correctly
        if not ret or img is None:
            print("Error: Failed to capture image.")
            continue

        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, conf = recognizer.predict(gray[y:y + h, x: x + w])
            profile = getProfile(id)
            print(profile)
            if profile is not None:
                cv2.putText(img, "Name: " + str(profile[1]), (x, y + h + 20), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 127), 2)
                cv2.putText(img, "Age: " + str(profile[2]), (x, y + h + 45), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 127),
                            2)

        # Display the frame with detected faces
        cv2.imshow("FACE", img)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
except KeyboardInterrupt:
    print("\nProgram interrupted by user.")
finally:
    # Release the camera and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()
    print("Resources released, program terminated.")
