import cv2 as cv

capture = cv.VideoCapture(0)  # to open Camera

# accessing pretrained model
model = cv.CascadeClassifier("face_detector.xml")

while True:
    boolean, frame = capture.read()
    if boolean == True:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        coordinate_list = model.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3)

        for (x, y, w, h) in coordinate_list:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv.imshow("Face Detection", frame)

        if cv.waitKey(20) == ord('x'):
            break

capture.release()
cv.destroyAllWindows()