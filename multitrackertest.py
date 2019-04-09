import cv2
import numpy as np
import math

#motiondetetortest3.py

#Yury is gay, and Dustin is EPIC

cap = cv2.VideoCapture('ants3_Trim.mp4')#"movingcircles_Trim.mp4"
fgbg = cv2.createBackgroundSubtractorMOG2(999, detectShadows=True)
kernel = np.ones((11,11),np.uint8)
tracker = cv2.TrackerMOSSE_create()
multiTracker = cv2.MultiTracker_create()
bboxes = []

params = cv2.SimpleBlobDetector_Params()
    
params.minThreshold = 0
params.maxThreshold = 255
params.filterByArea = True
params.minArea = 100
params.maxArea = 100000
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.blobColor = 0
detector = cv2.SimpleBlobDetector_create(params)



blobdetect = False 


cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('width','image', 1080, 1080, nothing)
cv2.createTrackbar('height','image', 720, 720, nothing)
cv2.createTrackbar('x1','image',0,1000,nothing)
cv2.createTrackbar('y1','image',0, 1000,nothing)


while(1):
    
    ret, frame = cap.read()

 
  

    boxes = multiTracker.update(frame)


    #cv2.circle(frame, (rectangle[0], rectangle[1]), 5, (0,255,0))

    fake_frame = frame
    gui_frame = frame 

    width = cv2.getTrackbarPos('width', 'image')
    height = cv2.getTrackbarPos('height', 'image')
    x1 = cv2.getTrackbarPos('x1', 'image')
    y1 = cv2.getTrackbarPos('y1', 'image')
    
    resize = gui_frame[ y1:height, x1:width]

    cv2.imshow('image', resize)
    closing = cv2.morphologyEx(fake_frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)
    
    

    ret, thresh=  cv2.threshold(dilate,127,255,cv2.THRESH_BINARY)
    
    fgmask = fgbg.apply(thresh, None, -1)

    keypoints = detector.detect(fgmask)

    bboxes = []

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
            
    if (blobdetect == False):
        if (len(bboxes) > 0):
            blobdetect = True
            #ok = tracker.init(frame, bboxes[0])
            multiTracker.add(tracker, frame, bbox)
    else:
        #ok, box = tracker.update(frame)
        success, box = multiTracker.update(frame)
        cv2.rectangle(frame,(int(box[0][0]),int(box[0][1])),(int(box[0][0]) + int(box[0][2]),int(box[0][1]) + int(box[0][3])),(255,255,255),3)
        cv2.circle(frame, (int(box[0][0]+box[0][2]/2), int(box[0][1]+box[0][3]/2)), 2, (0,255,0))
        

        print(box)

        



        

    
            
    #print(bboxes, "fgbg")
    

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
