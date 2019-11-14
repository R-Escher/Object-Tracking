# Object-Tracking
Python OpenCV Object Tracking using SIFT, SURF and ORB.

Tracks a object in a video, using selected keypoint detector.

Program run examples:
--

SIFT with 2000 keypoints and 10% of good match:
python object_tracking.py sift 2000 0.1

SURF with standard keypoints and 30% of good match:
python object_tracking.py surf 0 0.3

ORB with 10000 keypoints and 80% of good match:
python object_tracking.py orb 10000 0.8
