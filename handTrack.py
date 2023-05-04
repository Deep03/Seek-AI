"""
Hand Tracking Module 
Name: Upendra Pant
Date :May 04 2023

"""


import cv2 as cv
import mediapipe as mp
import time


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
        lmlist=[]
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
    
if __name__== "__main__":
 main()
