"""
Hand Tracking Module 
Name: Upendra Pant
Date :May 04 2023

"""



import cv2 as cv
import mediapipe as mp
import time
import math
import os
lmlist = []
"""
handDetector class does:
    -Initialzes all the parametrs
    -draws landmarks and connections and then returns processed image
    -Lists all the landmarks values and returns the list back-
"""
class  handDetector():
    def __init__(self,static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands=max_num_hands
        self.model_complexity=model_complexity
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence


        self.mpHands=mp.solutions.hands
        self.hands = self.mpHands.Hands(self.static_image_mode,self.max_num_hands,
                                        self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw=mp.solutions.drawing_utils

    #Reads the image then draws landmark and connecting lines,finally returns processed image
    def findHands(self,img,draw=True, connections_color=(0, 255, 0)):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                 self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS, connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=connections_color, thickness=2, circle_radius=2))   
        return img
    
    #Makes the list of all landmark values(index,x value and y value) and returns the list
    def findPosition(self,img,handNo=0, draw=True):

        if self.results.multi_hand_landmarks:
            myHand=self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,cx,cy])
                if draw:
                  cv.circle(img, (cx, cy), 7, (0, 0, 255), cv.FILLED)
        return lmlist
"""
Main function does:
    -takes all output from findHands and findPosition functions
    -checks given codition
    -puts FPS value on screen
    -Displays final Image
"""


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    pTime=0
    cTime=0
    cap=cv.VideoCapture(0)
    detector = handDetector()
    while(True):
        success, img =cap.read()
        img = detector.findHands(img)
        lmlist=detector.findPosition(img)
        if len(lmlist)!=0:
         print(lmlist[4])
        if not success:
            break
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv.putText(img,str(int(fps)),(5,60),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)

        cv.imshow("Image",img)

        if cv.waitKey(1)==ord("q"):
            break
    cap.release()
    cv.destroyAllWindows()
    return lmlist


    
if __name__== "__main__":
    lmlist1 = main()
    wrist_arr = lmlist1[0]
    thumb1 = lmlist1[2]
    thumb2 = lmlist1[4]
    index1 = lmlist1[5]
    index2 = lmlist1[8]
    middle1 = lmlist1[9]
    middle2 = lmlist1[12]
    ring1 = lmlist1[13]
    ring2 = lmlist1[16]
    pinky1 = lmlist1[17]
    pinky2 = lmlist1[20]
    thumb_dist = distance(thumb1[1], thumb1[2], thumb2[1], thumb2[2])
    index_dist = distance(index1[1], index1[2], index2[1], index2[2])
    middle_dist = distance(middle1[1], middle1[2], middle2[1], middle2[2])
    ring_dist = distance(ring1[1], ring1[2], ring2[1], ring2[2])
    pinky_dist = distance(pinky1[1], pinky1[2], pinky2[1], pinky2[2])
    max_finger = max(thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist)
    if (max_finger == thumb_dist):
        with open("seek/index.txt", 'w+') as f:
            f.write("IT WORKED!!!!")
            f.close()
    else:
        print("FU****** YOUUUUU IDIOT")

