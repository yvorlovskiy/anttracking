import cv2
import numpy as np

def firstRectangleParams(box):
    return(int(box[0]),int(box[1]))

def secondRectangleParams(box):
    return(int(box[0]) + int(box[2]),int(box[1]) + int(box[3]))

def imageOperations(frame):
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)
    ret, thresh = cv2.threshold(dilate,127,255,cv2.THRESH_BINARY)
    return thresh
    
cap = cv2.VideoCapture('ants3_Trim.mp4')
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

while(1):
    
    ret, frame = cap.read()
    
    fgmask = fgbg.apply(imageOperations(frame), None, -1)

    keypoints = detector.detect(fgmask)

    bboxes = []

    if (blobdetect == False):
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


    ok, box = tracker.update(frame)
    
    if (blobdetect == False):
        if (len(bboxes) == 2):
            blobdetect = True
            ok = tracker.init(frame, bboxes[0])
    else:
        cv2.rectangle(frame,firstRectangleParams(box), secondRectangleParams(box),(255,0,255),3)
        cv2.circle(frame, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 2, (255,255,255))

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

    
