#!/usr/bin/python
import numpy as np
import cv2
import sys
from skimage.measure import structural_similarity as ssim

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def get_avg_color(frame):
    avg_color_per_row = np.average(frame, axis=0)
    return np.average(avg_color_per_row, axis=0)

if sys.argv[1:]:
    filename = sys.argv[1]
else:
    print("Usage: %s <filename>" % sys.argv[0])
    exit(0)

cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
print("width="+str(width))
print("height="+str(height))
cnt=0

cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, 1000 * 30)

if(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(gray, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    height4 = int(height/4)
    width4 = int(width/4)

    frame_top = frame[0:(height/2), 0:width]
    frame_bottom = frame[(height/2):height, 0:width]
    frame_left = frame[0:height, 0:(width/2)].copy()
    frame_right = frame[0:height, (width/2):width].copy()
    #frame_tb = cv2.absdiff(frame_top,frame_bottom)
    #frame_lr = cv2.absdiff(frame_left,frame_right)

    avgtb = abs(get_avg_color(frame_top)) - abs(get_avg_color(frame_bottom))
    avglr = abs(get_avg_color(frame_left)) - abs(get_avg_color(frame_right))

    #avgtb = avgtb[0] + avgtb[1] + avgtb[2]
    #avglr = avglr[0] + avglr[1] + avglr[2]

    cnt = cnt + 1
    if (cnt > 0):
        m1 = mse(frame_top, frame_bottom)
        m2 = mse(frame_left, frame_right)
        #s = ssim(frame_top, frame_bottom)
        #hheight, hwidth = frame_top.shape[:2]
        #vheight, vwidth = frame_left.shape[:2]
        #print(str(hheight)+" "+str(hwidth) +" "+str(m1)+" "+str(m2)+"\n")
        if m1 < m2:
            print("video is OU")
        else:
            print("video is SBS")
        cnt = 0
    #cv2.imshow('frame',frame)
    #cv2.imshow('frame',gray)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cap.release()
cv2.destroyAllWindows()
