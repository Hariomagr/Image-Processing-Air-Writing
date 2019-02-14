import cv2
import imutils
import numpy as np
import math
bg=None
aWeight = 0.1
camera = cv2.VideoCapture(0)
top, right, bottom, left = 10, 400, 250, 680
num_frames = 0
points=[]
tempoints=None
flag=1
while(True):
    (grabbed, frame) = camera.read()
    frame = imutils.resize(frame, width=700)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    (height, width) = frame.shape[:2]
    roi = frame[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    keypress = cv2.waitKey(1) & 0xFF
    if num_frames < 30:
        if bg is None:
            bg = gray.copy().astype("float")
        else:
            cv2.accumulateWeighted(gray, bg, aWeight)
    else:
        diff = cv2.absdiff(bg.astype("uint8"), gray)
        thresholded = cv2.threshold(diff,25,255,cv2.THRESH_BINARY)[1]
        (_, cnts, _) = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if(len(cnts)==0):
            hand=None
            points=[]
            tempoints=None
        else:
            segmented = max(cnts, key=cv2.contourArea)
            hand=(thresholded,segmented)
        if(hand is not None):
            (thresholded, segmented) = hand
            cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 255),5)
            hull = cv2.convexHull(segmented)
            extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
            if(keypress == ord("p")):
                if(flag==0):
                    flag=1
                else:
                    flag=0
            if(flag==0):
                tempoints=None
                points.append(extreme_top)
            else:
                tempoints=extreme_top
            for i in points:
                (a,b)=i
                cv2.line(clone,(a+right,b+top),(a+right,b+top),(255,255,0),5)
            if(tempoints is not None):
                (a,b)=tempoints
                cv2.line(clone,(a+right,b+top),(a+right,b+top),(255,255,255),5)
            ROI=clone[top:bottom,10:290]
            if(keypress==ord("c")):
                cv2.imwrite("text.jpg",ROI)
            cv2.imshow("Thesholded", thresholded)
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    num_frames += 1
    cv2.imshow("Video Feed", clone)
    
    if keypress == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
