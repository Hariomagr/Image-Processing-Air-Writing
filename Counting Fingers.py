import cv2
import imutils
import numpy as np
import math
bg=None
aWeight = 0.1
camera = cv2.VideoCapture(0)
top, right, bottom, left = 10, 400, 250, 680
num_frames = 0
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
        else:
            segmented = max(cnts, key=cv2.contourArea)
            hand=(thresholded,segmented)
        if(hand is not None):
            (thresholded, segmented) = hand
            cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 0, 255),5)
            cv2.imshow("Thesholded", thresholded)
            hull = cv2.convexHull(segmented, returnPoints=False)
            defects = cv2.convexityDefects(segmented, hull)
            count_defects=0
            if(defects is not None):
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(segmented[s][0])
                    end = tuple(segmented[e][0])
                    far = tuple(segmented[f][0])
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                    if angle <= 90:
                        count_defects += 1
                cv2.putText(clone, str(count_defects+1), (220,bottom+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            else:
                cv2.putText(clone, "1", (220,bottom+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(clone, "Total fingers:", (10,bottom+100), cv2.FONT_HERSHEY_SIMPLEX, 1, (35,114,234), 2)
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    num_frames += 1
    cv2.imshow("Video Feed", clone)
    
    if keypress == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
