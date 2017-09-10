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

# doesn't seem to produce improvement... actually, i think it hurts 
def toLogPolar(img):
    height, width = img[:2]
    #scale = self.imgsize / math.log(self.imgsize)

    #convert to color, else logpolar crashes
    clr = cv2.cv.cvCreateImage(cv2.cv.cvSize(width, height), 8, 3);
    cv2.cv.cvCvtColor(img, clr, cv2.cv.CV_GRAY2RGB)

    dst = cv.cvCreateImage(cv.cvSize(width, height), 8, 3);
    cv2.cv.cvLogPolar(clr, dst, 
                  cv2.cv.cvPoint2D32f(width / 2, height/ 2), 
                  1, cv2.cv.CV_WARP_FILL_OUTLIERS)

    return dst

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

while(cap.isOpened()):
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
    
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    #frame_tb = cv2.absdiff(frame_top,frame_bottom)
    #frame_lr = cv2.absdiff(frame_left,frame_right)

    cv2.LogPolar(frame_left, dst, (x, y), 40, cv.CV_INTER_LINEAR + cv.CV_WARP_FILL_OUTLIERS)
    cv2.imshow('frame',dst)
    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
