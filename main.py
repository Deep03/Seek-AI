import numpy as np
import cv2 as cv
import os, time

# Open the camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Initialize frame counter
frame_count = 0

# Get the current working directory
cwd = os.getcwd()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Save the frame as an image
    filename = f"frame-imgs/frame_{frame_count}.jpg"
    path_frame = os.path.join(cwd, filename)
    cv.imwrite(path_frame, gray)

    # Update frame counter
    frame_count += 1
    
    # Display the resulting frame
    cv.imshow('frame', gray)
    
    # Exit on 'q' key press
    if cv.waitKey(1) == ord('q'):
        break
    

# Release the capture and close the window
cap.release()
cv.destroyAllWindows()

time.sleep(2)

# Ask the user if they want to keep the frames
response = input("Do you want to keep those frames? (y/n): ")
if response.upper() == 'Y':
    exit()
elif response.upper() == 'N':
    file_path = cwd + "/frame-imgs"

    # Get a list of all files in the folder
    files = os.listdir(file_path)

    # Loop through each file and delete it
    for file in files:
        # Delete the file
        os.remove(file_path + "/" + file)

    print("All files in the folder have been deleted.")