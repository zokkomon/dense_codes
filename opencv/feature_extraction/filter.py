import numpy
import cv2

Preview = 0
Blur = 1
Canny = 2
Sobel = 3
Laplacian = 4
Shi_to_Masi = 5
Harris = 6
SIFT = 7
FAST = 8
ORB = 9
BRIEF= 10
AT= 11

feature_params = dict(maxCorners=500,
                    qualityLevel=0.2,
                    minDistance=15,
                    blockSize=9)

filter = Preview
source = cv2.VideoCapture(0)

alive = True

while alive:
    has_frame,frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame,1)

    if filter==Preview:
        result = frame
    elif filter==Blur:
        result = cv2.blur(frame,(5,5))
    elif filter==Canny:
        result = cv2.Canny(frame,90,150)
    elif filter==Sobel:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.Sobel(frame_gray,cv2.CV_64F,1,0,ksize=3)
    elif filter==Laplacian:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.Laplacian(frame_gray,cv2.CV_64F)
    elif filter==Shi_to_Masi:
        result=frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray ,**feature_params)
        if corners is not None:
            for x,y in numpy.int32(corners).reshape(-1,2):
                cv2.circle(result,(x,y),10,(0,255,0),1)
    elif filter==Harris:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = cv2.cornerHarris(numpy.float32(frame_gray), 2, 5, 0.07)
        result = cv2.dilate(result,None)
    elif filter==SIFT:
        result = frame
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp,des = cv2.SIFT_create().detectAndCompute(frame_gray,None)
        cv2.drawKeypoints(frame_gray, kp, result, (255,0,0), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif filter==FAST:
        result = frame
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp = cv2.FastFeatureDetector_create().detect(frame_gray,None)
        cv2.drawKeypoints(frame_gray, kp, frame, (0,0,255), flags= cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    elif filter==ORB:
        result = frame
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kp,des = cv2.ORB_create().detectAndCompute(frame_gray,None)
        cv2.drawKeypoints(frame_gray, kp, result, (0,255,255), flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    elif filter==BRIEF:
        result = frame
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        star= cv2.xfeatures2d.StarDetector_create()
        brief =cv2.xfeatures2d.BriefDescriptorExtractor_create()
        kp = star.detect(frame_gray,None)
        kp,res = brief.compute(frame_gray,kp)
        # print(brief.descriptorSize())
        # print(res.shape)
        cv2.drawKeypoints(frame_gray, kp, result, (0,255,255), flags= 0)
    elif filter==AT:
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        result = cv2.adaptiveThreshold(frame_gray,80,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    cv2.imshow('filter',result)

    key = cv2.waitKey(1)
    if key==ord('Q') or key==ord('q'):
        alive= False
    elif key==ord('U') or key==ord('u'):
        filter=Blur
    elif key==ord('C') or key==ord('c'):
        filter=Canny
    elif key==ord('E') or key==ord('e'):
        filter=Sobel
    elif key==ord('L') or key==ord('l'):
        filter=Laplacian
    elif key==ord('M') or key==ord('m'):
        filter=Shi_to_Masi
    elif key==ord('H') or key==ord('h'):
        filter=Harris
    elif key==ord('S') or key==ord('s'):
        filter=SIFT
    elif key==ord('F') or key==ord('f'):
        filter=FAST
    elif key==ord('O') or key==ord('o'):
        filter=ORB
    elif key==ord('B') or key==ord('b'):
        filter=BRIEF
    elif key==ord('P') or key==ord('p'):
        filter=Preview
    elif key==ord('A') or key==ord('a'):
        filter=AT

source.release()
cv2.destroyWindow('filter')




