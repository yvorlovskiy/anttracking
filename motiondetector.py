import cv2
import numpy as np

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
print(rectangle)'''

i = 0

blobdetect = False

while(1):
    i =  i + 1
    ret, frame = cap.read()


    #cv2.circle(frame, (rectangle[0], rectangle[1]), 5, (0,255,0))

    fake_frame = frame
    closing = cv2.morphologyEx(fake_frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)

    ret, thresh=  cv2.threshold(dilate,127,255,cv2.THRESH_BINARY)
    
    fgmask = fgbg.apply(thresh, None, -1)#thresh

    keypoints = detector.detect(fgmask)

    bboxes = []

    for keypoint in keypoints:
        if keypoint:
            x = int(round(keypoint.pt[0]))
            y = int(round(keypoint.pt[1]))
            s = int(round(keypoint.size))
            #cv2.circle(frame, (x, y), 20, (0,255,0))
            bbox = (x-50,y-50,s*5,s*5)
            bboxes.append(bbox)
            #cv2.rectangle(fgmask,(x-s,y+s),(x+s,y-s),(255,255,255),3)
            #cv2.circle(frame, (x-s, y-s), 5, (0,255,0))
                #ok = tracker.init(frame, bbox)

                
                
    ok, box = tracker.update(frame)
    
    #print(len(bboxes))

    if (True):            
        if (blobdetect == True):
            #print("great")
            ok, box = tracker.update(frame)
            cv2.rectangle(frame,(int(box[0]),int(box[1])),(int(box[0]) + int(box[2]),int(box[1]) + int(box[3])),(255,0,255),3)
            #cv2.rectangle(frame,(bboxes[0][0]-bboxes[0][2],bboxes[0][1]+bboxes[0][3]),(bboxes[0][0]+bboxes[0][2],bboxes[0][1]-bboxes[0][3]),(255,255,255),3)
            #cv2.rectangle(frame,(int(bbbox[0]),int(bbbox[1])),(int(bbbox[0]) + int(bbbox[2]),int(bbbox[1]) + int(bbbox[3])),(0,255,255),3)
            print(box, "tracker")
            #print(bboxes[0], "bbox")
            #print(bbbox)
           
        else:
            if (len(bboxes) == 1):
                #print("of course")
                blobdetect = True
                ok = tracker.init(frame, bboxes[0])#(350, 626, 400, 676)
                bbbox = bboxes[0]
        if (len(bboxes)=0):

            
    
    #print(blobdetect)
            #if (i == 60):
                #multiTracker.add(tracker, frame, (x-50, y+50, 50, 50))

    #success, boxes = multiTracker.update(frame)

    '''if (len(boxes) >= 4):
        for box in boxes:
            cv2.rectangle(frame,(boxes[0],boxes[1]),(boxes[0] + boxes[2],boxes[1] + boxes[3]),(0,0,255),3)'''

            
    #print(bboxes, "fgbg")
    
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('frame', frame)
    
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
