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

    radius= 0.37*W
    
    return Y,X,radius


#Everything below T1 is background, everything above T2 is colony
T1 = 60
T2 = 100

# Setup SimpleBlobDetector parameters.
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

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

for file in file_names:
    #Read the image images
    img = cv2.imread(folder_path+'/'+file)

    #Define the center and the radius of the ROI
    X_cent, Y_cent, Rad = define_circular_ROI(img)

    #Set everything outside of the ROI to 0
    H,W = img.shape[:2]
    mask = create_circular_mask(H, W, (X_cent, Y_cent), Rad)
    img[~mask] = 0

	# Detect blobs.
    keypnts = detector.detect(img)
    print('{} has {} blobs'.format(file.split('.')[0], len(keypnts)))

	# Draw detected blobs as red circles.
    im_with_keys = cv2.drawKeypoints(img, keypnts, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
    #Write the final image
    cv2.imwrite(processed_path+'/'+file, im_with_keys)