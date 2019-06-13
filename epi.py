import cv2
import numpy as np
from matplotlib import pyplot as plt


MIN_MATCH_COUNT = 10

img1 = cv2.imread('/Users/rajatgirhotra/Desktop/marker2.jpg',0)  #queryimage # left image
img2 = cv2.imread('/Users/rajatgirhotra/Desktop/marker.jpg',0) #trainimage # right image


sift = cv2.xfeatures2d.SIFT_create()

#freakExtractor = cv2.xfeatures2d.FREAK_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)



# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []

# ratio test as per Lowe's paper
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    
    #out = cv2.warpPerspective(img1,img2,M, (800,800))
    cv2.imwrite("out1.png",out)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape[:3]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()




#def drawlines(img1,img2,lines,pts1,pts2):
#    ''' img1 - image on which we draw the epilines for the points in img2
#        lines - corresponding epilines '''
#    r,c = img1.shape
#    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
#    for r,pt1,pt2 in zip(lines,pts1,pts2):
#        color = tuple(np.random.randint(0,255,3).tolist())
#        x0,y0 = map(int, [0, -r[2]/r[1] ])
#        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#    return img1,img2

# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image


#lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,-1,2),2,F)
#lines1 = lines1.reshape(-1,3)
#img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image

#lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,-1,2), 1,F)
#lines2 = lines2.reshape(-1,3)
#img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

#plt.subplot(121),plt.imshow(img5)
#plt.subplot(122),plt.imshow(img3)
#plt.show()
