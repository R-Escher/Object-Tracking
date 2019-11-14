import tkinter
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

matchPercent = 0.05


# creates SIFT
sift = cv.xfeatures2d.SIFT_create()


# loads train image into grayscale
train_Img = cv.imread('book_train.jpg', cv.IMREAD_GRAYSCALE)


# loads test video
cap = cv.VideoCapture('book.mp4')


# resizes images to fit screen
train_Img =  cv.resize(train_Img, (800, 1000), interpolation = cv.INTER_AREA)
#test_Img =  cv.resize(test_Img,(1600,2200), interpolation = cv.INTER_AREA)


# detects and computes train images
print("Computing train image...")
train_KP, train_Desc = sift.detectAndCompute(train_Img,None)


# declares use Brute-Force Matcher
bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
cont = 0
plt.show()

while True:

    # loads frame from video
    _, test_Img = cap.read()
    #test_Img = cv.cvtColor(test_Img, cv.COLOR_BGR2GRAY)
    
    # converts from bgr to rgb due to mpl bug
    test_Img = cv.cvtColor(test_Img, cv.COLOR_BGR2RGB)
    test_Img =  cv.resize(test_Img, (1200, 700), interpolation = cv.INTER_AREA)

    # detects and computes test images
    print("Computing video image...")
    test_KP, test_Desc = sift.detectAndCompute(test_Img,None)

    # find matches between train and test images using KNN
    print ("Finding matches with Matcher...")
    matches = bf.match(test_Desc, train_Desc)
    matches = sorted(matches, key = lambda x:x.distance)

    # select best matches
    numGoodMatches = int(len(matches) * matchPercent)
    matches = matches[:numGoodMatches]

    query_pts = np.float32([test_KP[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    train_pts = np.float32([train_KP[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    h, mask = cv.findHomography( train_pts, query_pts, cv.RANSAC)        
    matches_mask = mask.ravel().tolist()

    # Use homography
    height, width = train_Img.shape
    pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, h)
    #im1Reg = cv.warpPerspective(test_Img, h, (width, height))    

    homography = cv.polylines(test_Img, [np.int32(dst)], True, (255, 0, 0), 3)

    # Draw best matches.
    img3 = cv.drawMatches( test_Img, test_KP,train_Img, train_KP, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(1); plt.clf()
    plt.imshow(homography)
    plt.imshow(img3)
    plt.pause(0.0000001)

    print(cont)
    cont=cont+1


plt.close()
cap.release(0)
cv.destroyAllWindows()