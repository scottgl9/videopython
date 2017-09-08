#!/usr/bin/python
import numpy as np
import cv2
import numpy as np
import cv2
import sys

def get_avg_color(frame):
    avg_color_per_row = np.average(frame, axis=0)
    return np.average(avg_color_per_row, axis=0)

if sys.argv[1:]:
    filename = sys.argv[1]
else:
    print("Usage: %s <filename>" % sys.argv[0])
    exit(0)

print("OpenCV Version: " + cv2.__version__)
cap = cv2.VideoCapture(filename)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("width="+str(width))
print("height="+str(height))
cnt=0

#pts3d = pts3d.astype('float32')
#pts2d = pts2d.astype('float32')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(gray, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    height4 = int(height/4)
    width4 = int(width/4)

    #camera_matrix = cv2.initCameraMatrix2D(([pts3d],[pts2d]), frame.shape[:2])

    #cv2.calibrateCamera([pts3d], [pts2d], (width, height), camera_matrix, None,
    #                flags=cv2.CALIB_USE_INTRINSIC_GUESS)


    frame_top = frame[0:(height/2), 0:width]
    frame_bottom = frame[(height/2):height, 0:width]
    frame_left = frame[0:height, 0:(width/2)].copy()
    frame_right = frame[0:height, (width/2):width].copy()
    #frame_tb = cv2.absdiff(frame_top,frame_bottom)
    #frame_lr = cv2.absdiff(frame_left,frame_right)
    img = frame_left
    res = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    #dst = cv2.fastNlMeansDenoisingColored(res,None,10,10,7,21)
    gaussian_3 = cv2.GaussianBlur(img, (9,9), 10.0)
    dst = cv2.addWeighted(img, 1.5, gaussian_3, -0.5, 0, img)

    #pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    #pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    #M = cv2.getPerspectiveTransform(pts1,pts2)
    #dst = cv2.warpPerspective(res,M,(res.shape[0],res.shape[1]))
    #img2 = cv2.logPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)
    #img3 = cv2.linearPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, cv2.WARP_FILL_OUTLIERS)
    #dst = cv2.fisheye.distortPoints(img, np.eye(3), None)
    cv2.imshow('frame', dst)

    #avgtb = np.uint8(get_avg_color(frame_tb))
    #avglr = np.uint8(get_avg_color(frame_lr))

    #avgtb = avgtb[0] + avgtb[1] + avgtb[2]
    #avglr = avglr[0] + avglr[1] + avglr[2]

    #cnt = cnt + 1
    #if (cnt > 60):
    #    hheight, hwidth = frame_top.shape[:2]
    #    vheight, vwidth = frame_left.shape[:2]
    #    print(str(hheight)+" "+str(hwidth) +" "+str(vheight)+" "+str(vwidth)+"\n")
    #    cnt = 0

    #cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
