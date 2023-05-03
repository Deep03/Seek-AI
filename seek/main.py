import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break  # break out of the loop if 'q' is pressed

# Release the capture and close the window
cap.release()
# print(f"The type of ret is {type(ret)}")
# print(f"The type of cv is {type(cv)}")
# print(f"The type of cap is {type(cap)}")
# print(f"The type of frame is {type(frame)}")
cv.destroyAllWindows()
