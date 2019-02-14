import pyautogui
import time
import cv2
import imutils
import numpy as np
import math
pyautogui.FAILSAFE = True

def up():
    pyautogui.keyDown('up')
    pyautogui.keyUp('up')
    #time.sleep(0.005)
    print('Jump control')
def duck():
    pyautogui.keyDown('down')
    #time.sleep(0.05)
    pyautogui.keyUp('down')
    print('Duck control')
def normal():
     pyautogui.keyUp('down')
     pyautogui.keyUp('space')
     #time.sleep(0.005)
     print('Normal')
bg=None
aWeight = 0.1
camera = cv2.VideoCapture(0)
top, right, bottom, left = 10, 400, 250, 680
num_frames = 0
avgy=int((top+bottom)/2)
current=0
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
            hull = cv2.convexHull(segmented)
            extreme_top = tuple(hull[hull[:, :, 1].argmin()][0])
            (a,b)=extreme_top
            if(avgy+40<b):
                if(current!=-1):
                    duck()
                    current=-1
            elif(avgy-40>b):
                if(current!=1):
                    up()
                    current=1
            elif(current!=0):
                normal()
                current=0
            cv2.circle(clone,(a+right,b+10),2,(255,255,255),5)
    cv2.line(clone,(left,avgy),(right,avgy),(0,255,255),2)
    cv2.line(clone,(left,avgy+40),(right,avgy+40),(255,255,255),2)
    cv2.line(clone,(left,avgy-40),(right,avgy-40),(255,255,255),2)
    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    num_frames += 1
    cv2.imshow("Video Feed", clone)
    if keypress == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
