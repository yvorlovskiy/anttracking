import numpy
import cv2

cap = cv2.VideoCapture('multi_ants2.mov')


while(1):
    ret, frame = cap.read()

    noisy = cv2.fastNlMeansDenoisingMulti(frame, 2, 5, None, 4, 7, 35)


    cv2.imshow('noisy', noisy)
    cv2.imshow('frame', frame)    
    k = cv2.waitKey(30) & 0xff
    
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()