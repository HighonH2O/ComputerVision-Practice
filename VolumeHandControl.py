import cv2
import mediapipe as mp
import numpy as np
import time
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


#######################################
wCam , hCam = 640 , 480
#######################################

# Initialise camera
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4,hCam)

#Initialise hand detector
pTime = 0
detector = htm.handDetector(detectionCon = 0.7)

#Initialise Audio Utilities
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

# devices = AudioUtilities.GetSpeakers()
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    img = detector.findHands(img)
    lmList = detector.findPosition(img , draw = False)

    if len(lmList)!=0:
        # print(lmList[4], lmList[8])

        x1 ,y1 = lmList[4][1] , lmList[4][2]
        x2 , y2 = lmList[8][1] , lmList[8][2]
        cx ,cy = (x1+x2) // 2 , (y1+y2) // 2

        cv2.circle(img , (x1,y1) , 7 , (255,0,255), cv2.FILLED)
        cv2.circle(img , (x2,y2) , 7 , (255,0,255), cv2.FILLED)
        cv2.line(img , (x1,y1) , (x2,y2) , (255,0,255) , 3)
        cv2.circle(img , (cx,cy) , 7 , (255,0,255), cv2.FILLED)

        length = math.hypot(x2-x1 , y2-y1)
        # print(length)

        # Hand range - 15 - 190
        # Vol range - -37 to -4

        vol = np.interp(length , [15,190] , [minVol, maxVol])  #Need to tweak the values , maxvol til 50 going
        print(int(length) , vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<25:
            cv2.circle(img , (cx,cy), 7 , (0,255,0) , cv2.FILLED)









    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img , f'FPS:{int(fps)}', (40,70), cv2.FONT_HERSHEY_PLAIN,3,(255,0,0) , 3)

    cv2.imshow("Img" , img)
    cv2.waitKey(1)