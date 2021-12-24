import matplotlib as plt
import numpy as np
import cv2
import os
from skimage import data, filters

#get a list of files in the folder with pics
from os import listdir
from os.path import isfile, join
folder_path = os.path.dirname(__file__)+'/Smpl_Im'
processed_path = os.path.dirname(__file__)+'/Thresholded'
file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

# finction creates a circular mask based on the dimentions, radius and the desired loaction
#I modified it from here:
#https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    #Compute square of the radius to avoid computing sqrt on every step
    radius_sq = radius**2

    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (X - center[0])**2 + (Y-center[1])**2

    base_cond = dist_from_center_sq <= radius_sq
    mask = np.array(np.where(base_cond, 255, 0), dtype=np.uint8)

    return mask

def define_circular_ROI(image):
    
    H, W = image.shape[:2]
    Y = H/2
    X = W/2

    radius= 0.37*W
    
    return Y,X,radius

def Setup_Blob_Detector():

    params = cv2.SimpleBlobDetector_Params()
 
    #Change thresholds
    params.minThreshold = 40
    params.maxThreshold = 150
    params.thresholdStep = 3
    params.minDistBetweenBlobs = 5
    params.minRepeatability = 7


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 250
    params.maxArea = 5000000

    # Filter by CircularityTrue
    params.filterByCircularity = True
    params.minCircularity = 0.25
    params.maxCircularity = 0.99

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.2
    params.maxConvexity = 0.99

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    params.maxInertiaRatio = 0.99

    # Filter by Color
    params.filterByColor = True
    params.blobColor = 255

    return params

def alignImages(im1, im2, msk, name):
    #I poached this code from https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, mask=msk)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, mask=msk)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite(name, imMatches)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    if len(points1) >=4 and len(points2)>=4:
        h, mask_1 = cv2.findHomography(points1, points2, cv2.RANSAC,ransacReprojThreshold=5.0)
    else:
        return im1, []

    if h is not None:

        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.Perspective(im1, h, (height, width), np.array([]))

        return im1Reg, h

    else:
        return im1, h


# Create a detector with the parameters
params = Setup_Blob_Detector()
blob_detector = cv2.SimpleBlobDetector_create(params)

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Read reference image
refFilename = folder_path+'/'+'Ref.jpg'
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

for file in file_names:
    #Read the image
    img = cv2.imread(folder_path+'/'+file)

    #Define the center and the radius of the ROI
    X_cent, Y_cent, Rad = define_circular_ROI(img)

    #Set everything outside of the ROI to 0
    H,W = img.shape[:2]
    mask = create_circular_mask(H, W, (X_cent, Y_cent), Rad)

    print("Aligning images ...")

    # Registered image will be sotred in imReg.
    # The estimated homography will be stored in h.
    imReg, h = alignImages(img, imReference, mask, file)

	# Detect blobs.
    keypnts = blob_detector.detect(imReg)
    print('{} has {} blobs'.format(file.split('.')[0], len(keypnts)))

	# Draw detected blobs as red circles.
    im_with_keys = cv2.drawKeypoints(imReg, keypnts, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
    #Write the final image
    cv2.imwrite(processed_path+'/'+file, im_with_keys)