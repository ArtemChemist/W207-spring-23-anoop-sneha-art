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

    mask = dist_from_center_sq <= radius_sq
    return mask

def define_circular_ROI(image):
    
    H, W = image.shape[:2]
    Y = H/2
    X = W/2

    radius= 0.34*W
    
    return Y,X,radius


#Everything below T1 is background, everything above T2 is colony
T1 = 60
T2 = 100

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

#Change thresholds
params.minThreshold = 10
params.maxThreshold = 250
params.thresholdStep = 10
#params.minDistBetweenBlobs = 10

# Filter by Area.
#params.filterByArea = True
#params.minArea = 1500

# Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.1

# Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87

# Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01

# Create a detector with the parameters
# OLD: detector = cv2.SimpleBlobDetector(params)
detector = cv2.SimpleBlobDetector_create(params)

for file in file_names:
    #Read the image images
    img = cv2.imread(folder_path+'/'+file)

    #Conver to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Filter out dust and pepper
    img_blur = cv2.medianBlur(img_gray,1)

    #Define the center and the radius of the ROI
    X_cent, Y_cent, Rad = define_circular_ROI(img_blur)

    #Set everything outside of the ROI to 0
    H,W = img_blur.shape[:2]
    mask = create_circular_mask(H, W, (X_cent, Y_cent), Rad)
    img_blur[~mask] = 0

	# Detect blobs.
    keypnts = detector.detect(img_blur)
    print('{} has {} blobs'.format(file.split('.')[0], len(keypnts)))

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keys = cv2.drawKeypoints(img, keypnts, np.array([]), (0,255,0), cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
	
    #Write the final image
    cv2.imwrite(processed_path+'/'+file, im_with_keys)