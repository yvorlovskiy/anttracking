import cv2
import numpy as np
import math

#motiondetetortest3.py

#Dustin is EPIC

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

'''ret, frame = cap.read()
rectangle = cv2.selectROI(frame)
'''
 

blobdetect = False 



while(1):
    
    ret, frame = cap.read()

    #boxes = multiTracker.update(frame)


    #cv2.circle(frame, (rectangle[0], rectangle[1]), 5, (0,255,0))

    fake_frame = frame
    closing = cv2.morphologyEx(fake_frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)

    ret, thresh=  cv2.threshold(dilate,127,255,cv2.THRESH_BINARY)
    
    fgmask = fgbg.apply(thresh, None, -1)

    keypoints = detector.detect(fgmask)

    bboxes = []

    for keypoint in keypoints:
        if keypoint:
            x = int(round(keypoint.pt[0]))
            y = int(round(keypoint.pt[1]))
            s = int(round(keypoint.size))/2
            bbox = (x-100,y-100,100,100)
            bboxes.append(bbox)
            cv2.rectangle(fgmask, (x-100,y+100),(x+100,y-100),(255,255,255),3)
            cv2.circle(frame, (int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2)), 2, (255,255,255))
            
    if (blobdetect == False):
        if (len(bboxes) > 0):
            blobdetect = True
            ok = tracker.init(frame, bboxes[0])
    else:
        ok, box = tracker.update(frame)
        cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[0]) + int(box[2]),int(box[1]) + int(box[3])),(255,255,255),3)
        cv2.circle(frame, (int(box[0]+box[2]/2), int(box[1]+box[3]/2)), 2, (255,255,255))
        
        print(box)

    ''' for bbox in bboxes:
            
            #multiTracker.add(frame)
            
            for box in boxes:
                
                x1 = (box[0]+box[2]/2)
                x2 = (bbox[0]+bbox[2]/2)
                y1 = (box[1]+box[3]/2)
                y2 = bbox[1]+bbox[3]/2
                print(math.hypot(int(x2-x1), int(y2-y1)))'''
    
            

       

        

    
            
    print(bboxes, "fgbg")
    

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
