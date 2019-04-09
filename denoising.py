    1 import numpy as np
    2 import cv2
    3 from matplotlib import pyplot as plt
    4 
    5 cap = cv2.VideoCapture('multi_ants.mov')
    6 
    7 # create a list of first 5 frames
    8 img = [cap.read()[1] for i in xrange(5)]
    9 
   10 # convert all to grayscale
   11 gray = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in img]
   12 
   13 # convert all to float64
   14 gray = [np.float64(i) for i in gray]
   15 
   16 # create a noise of variance 25
   17 noise = np.random.randn(*gray[1].shape)*10
   18 
   19 # Add this noise to images
   20 noisy = [i+noise for i in gray]
   21 
   22 # Convert back to uint8
   23 noisy = [np.uint8(np.clip(i,0,255)) for i in noisy]
   24 
   25 # Denoise 3rd frame considering all the 5 frames
   26 dst = cv2.fastNlMeansDenoisingMulti(noisy, 2, 5, None, 4, 7, 35)
   27 
   28 plt.subplot(131),plt.imshow(gray[2],'gray')
   29 plt.subplot(132),plt.imshow(noisy[2],'gray')
   30 plt.subplot(133),plt.imshow(dst,'gray')
   31 plt.show()