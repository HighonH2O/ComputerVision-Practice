import time
import os
import cv2
import HandTrackingModule as htm

wCam , hCam = 640,480

cap = cv2.VideoCapture(0)

cap.set(3,wCam)
cap.set(4,hCam)

folderpath = "D:\Desktop\PycharmProject\HandTrackingProject\Sample Finger Images"
myList = os.listdir(folderpath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderpath}/{imPath}')
    # print(f'{folderpath}/{imPath}')
    overlayList.append(image)


print(len(overlayList)) 

detector = htm.handDetector(detectionCon = 0.75)

pTime = 0

tipIds = [4, 8 , 12 , 16 , 20]

while True:
    success,img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img , draw = False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)


        #4 fingers
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)




    h , w, c = overlayList[0].shape

    img[0:200, 0:134] = overlayList[1]

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    #FPS is not showing on camera - fix later with knowing pixel dimensions of the camera.
    cv2.putText(img , f'FPS: {int(fps)}',(400,700), cv2.FONT_HERSHEY_PLAIN, 3 , (255,0,0) , 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

#Functional project - only need to add the image changing feature according to number of fingers and add some visual features (oo lazy)