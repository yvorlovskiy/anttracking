import cv2
import numpy as np

#tunnel with food


def ImageOperation(frame):
    closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(closing, (5, 5), iterations=1)
    ret, thresh =  cv2.threshold(dilate,127,255,cv2.THRESH_BINARY)
    return thresh

def GetBoundingBoxes(thresh):
    fgmask = BackgroundSubtrator.apply(thresh, None, -1)
    bboxes = []
    keypoints = BlobDetector.detect(fgmask)
    for keypoint in keypoints:
        if keypoint:
            x = int(round(keypoint.pt[0]))
            y = int(round(keypoint.pt[1]))
            s = int(round(keypoint.size))/2
            bbox = (x-s,y+s,x,y)
            bboxes.append(bbox)
            #cv2.rectangle(frame,(x-s,y+s),(x+s,y-s),(255,0,255),3)
            return bboxes
            




def AddTrackers(BoundingBoxes, TrackingBoxes):
    if (BoundingBoxes != None):
        Difference = len(BoundingBoxes) - len(TrackingBoxes)
        for i in range(Difference):
            multiTracker.add(Tracker, frame, BoundingBoxes[i])

cap = cv2.VideoCapture('ants3_Trim.mp4')
 
BackgroundSubtrator = cv2.createBackgroundSubtractorMOG2(999, detectShadows=True)
Parameters = cv2.SimpleBlobDetector_Params()
Tracker = cv2.TrackerMOSSE_create()
multiTracker = cv2.MultiTracker_create()

Parameters.minThreshold = 0
Parameters.maxThreshold = 255
Parameters.filterByArea = True
Parameters.minArea = 100
Parameters.maxArea = 100000
Parameters.filterByCircularity = False
Parameters.filterByInertia = False
Parameters.filterByConvexity = False
Parameters.filterByColor = False
Parameters.blobColor = 0
BlobDetector = cv2.SimpleBlobDetector_create(Parameters)

kernel = np.ones((11,11),np.uint8)

ret, frame = cap.read()

#multiTracker.add(Tracker, frame, (247, 393, 258, 382))

i = 0

while(1):
    ret, frame = cap.read()

    fake_frame = frame
    
    success, boxes = multiTracker.update(frame)

    bboxes = GetBoundingBoxes(ImageOperation(frame))
    print(AddTrackers(GetBoundingBoxes(ImageOperation(frame)), boxes))

    i = i + 1

    if (i == 3):
        multiTracker.add(Tracker, frame, (324, 609, 100,  80))
        

    if (i > 3):
        cv2.rectangle(fake_frame,(int(boxes[0][0]),int(boxes[0][1]), (int(boxes[0][0]+boxes[0][2]), int(boxes[0][1] + boxes[0][3])),(255,0,255),3))

    print(boxes)
    print(bboxes)

    
    cv2.imshow('fgmask', ImageOperation(frame))
    cv2.imshow('frame', frame)
    cv2.imshow('fake_frame', fake_frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

