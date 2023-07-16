import cv2 as cv
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime=0
cTime=0
cap=cv.VideoCapture(0)
detector = htm.handDetector()
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
