
import cv2
import numpy as np
import sqlite3

# Load the Haar Cascade file
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

def insertOrUpdate(Id, Name, Age):
    conn = sqlite3.connect("sqlite.db")  # connect to db
    cmd = "SELECT * FROM STUDENTS WHERE ID=" + str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if isRecordExist == 1:
        conn.execute("UPDATE STUDENTS SET Name=?, Age=? WHERE id=?", (Name, Age, Id))
    else:
        conn.execute("INSERT INTO STUDENTS (Id, Name, Age) VALUES (?, ?, ?)", (Id, Name, Age))
    conn.commit()
    conn.close()

# Collect user details
Id = input('Enter User Id: ')
name = input('Enter User Name: ')
age = input('Enter User Age: ')

# Insert or update user details in the database
insertOrUpdate(Id, name, age)

sampleNum = 0
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite("dataset/user." + str(Id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 20:
        break

cam.release()
cv2.destroyAllWindows()
