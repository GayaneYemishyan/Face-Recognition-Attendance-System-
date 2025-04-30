import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime

import csv

csv_file = r'C:\Users\User\OneDrive\Desktop\ProjectS2\Project\face_data.csv'

images = []
classNames = []

with open(csv_file, 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)

    for row in reader:
        name = row['Name']
        surname = row['Surname']
        photo_path = row['Photos']

        if os.path.exists(photo_path):
            img = cv2.imread(photo_path)
            if img is not None:
                images.append(img)
                classNames.append(f"{name} {surname}")  # Save full name
            else:
                print(f"Warning: Could not read image at {photo_path}")
        else:
            print(f"Warning: File does not exist at {photo_path}")

print(classNames)
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')

def cleanAttendanceFileKeepingHeader():
    try:
        with open('Attendance.csv', 'w', newline='') as f:
            f.write("Name,Time\n")
        print("Attendance.csv has been cleaned, and the header is now 'Name,Time'.")
    except FileNotFoundError:
        print("Attendance.csv not found.")


cleanAttendanceFileKeepingHeader()

encodeListKnown = findEncodings(images)
print('Encoding Completed')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    threshold = 0.6

    for encodeFace, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < threshold:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x1, y2, x2 = faceLoc
            y1, x1, y2, x2 = 4 * y1, 4 * x1, 4 * y2, 4 * x2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)
        else:
            print("No matches found")
            y1, x1, y2, x2 = faceLoc
            y1, x1, y2, x2 = 4 * y1, 4 * x1, 4 * y2, 4 * x2
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "UNKNOWN", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', img)

    cv2.waitKey(1)
