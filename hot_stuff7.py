import cv2
import numpy as np
import math

#motiondetetortest3.py

#Yury is EPIC, and Dustin is EPIC

cap = cv2.VideoCapture('multi_ants2.mov')#"movingcircles_Trim.mp4"'multi_ants2.mov'
fgbg = cv2.createBackgroundSubtractorMOG2(999, detectShadows=True)
kernel = np.ones((11,11),np.uint8)
tracker = cv2.TrackerMOSSE_create()
multiTracker = cv2.MultiTracker_create()
bboxes = []

params = cv2.SimpleBlobDetector_Params()
    
params.minThreshold = 0
params.maxThreshold = 30
params.filterByArea = True
params.minArea = 150
params.maxArea = 100000
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.blobColor = 0
detector = cv2.SimpleBlobDetector_create(params)



blobdetect = False 


while(1):
    
    ret, frame = cap.read()

    boxes = multiTracker.update(frame)


    #cv2.circle(frame, (rectangle[0], rectangle[1]), 5, (0,255,0))

    fake_frame = frame
    
    
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    #median = cv2.medianBlur(frame, 15)

    #noisy = cv2.fastNlMeansDenoisingMulti(frame, 2, 5, None, 4, 7, 35)

    
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)
    #smoothed = cv2.filter2D(dilate, -10, kernel)
    

    ret, thresh=  cv2.threshold(dilate,75,255,cv2.THRESH_BINARY)
    
    fgmask = fgbg.apply(thresh, None, -1)

    keypoints = detector.detect(fgmask)

    bboxes = []
    boxes = []


    for keypoint in keypoints:
        if keypoint:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = int(keypoint.size)/2
            bbox = (x-50,y-50,100,100)
            bboxes.append(bbox)
            cv2.rectangle(fgmask, (x-100,y+100),(x+100,y-100),(255,255,255),3)
            cv2.circle(frame, (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), 2, (255,255,255))
            cv2.circle(frame, (x, y), 5, (0,255,0))
            
    
    
            
    print(bboxes, "fgbg")
    

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


# NEXT - add and release multitrackers

# adding 