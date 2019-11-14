import tkinter
import numpy as np
import sys
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# argument parser
if len(sys.argv) == 5:
    video = sys.argv[1]
    kp_detector = sys.argv[2]
    numb_of_kp = int(sys.argv[3])
    matchPercent = float(sys.argv[4])

    if kp_detector == "sift":
        kp_detector = cv.xfeatures2d.SIFT_create(numb_of_kp)
        # sets BFM recommended configuration for sift
        bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)
    elif kp_detector == "surf":
        kp_detector = cv.xfeatures2d.SURF_create(numb_of_kp)
        # sets BFM recommended configuration for surf
        bf = cv.BFMatcher(cv.NORM_L2,crossCheck=True)        
    elif kp_detector == "orb":
        kp_detector = cv.ORB_create(numb_of_kp)
        # sets BFM recommended configuration for surf
        bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)           
else:
    print("Number of arguments must be 4:\n  1. Name and Format of video (example.mp4)\n  2. Type of KeyPoint Detector and Descriptor Extractor\n  3. Number of KeyPoints to be found (0 sets standard number) \n  4. Percentage of good matches to be considered")
    sys.exit()



# loads train image into grayscale
train_Img = cv.imread('book_train.jpg', cv.IMREAD_GRAYSCALE)


# loads test video
cap = cv.VideoCapture(str(video))


# resizes images to fit screen
train_Img =  cv.resize(train_Img, (800, 1000), interpolation = cv.INTER_AREA)
#test_Img =  cv.resize(test_Img,(1600,2200), interpolation = cv.INTER_AREA)


# detects and computes train images
print("Computing train image...")
train_KP, train_Desc = kp_detector.detectAndCompute(train_Img,None)


cont = 0
plt.show()

while True:

    # loads frame from video
    _, test_Img = cap.read()
    #test_Img = cv.cvtColor(test_Img, cv.COLOR_BGR2GRAY)
    
    # converts from bgr to rgb due to mpl bug
    #test_Img = cv.cvtColor(test_Img, cv.COLOR_BGR2RGB)
    test_Img = cv.cvtColor(test_Img, cv.COLOR_BGR2GRAY)
    test_Img =  cv.resize(test_Img, (2400, 1400), interpolation = cv.INTER_AREA)

    # detects and computes test images
    print("Computing video image...")
    test_KP, test_Desc = kp_detector.detectAndCompute(test_Img,None)

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
    h, _ = cv.findHomography( train_pts, query_pts, cv.RANSAC)        

    # Use homography
    height, width = train_Img.shape
    pts = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, h)
    #im1Reg = cv.warpPerspective(test_Img, h, (width, height))    

    homography = cv.polylines(test_Img, [np.int32(dst)], True, (255, 0, 0), 3)

    # Draw best matches.
    img3 = cv.drawMatches( homography, test_KP,train_Img, train_KP, matches, None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(1); plt.clf()
    plt.imshow(img3)
    plt.pause(0.0000001)

    print(cont)
    cont=cont+1


plt.close()
cap.release(0)
cv.destroyAllWindows()