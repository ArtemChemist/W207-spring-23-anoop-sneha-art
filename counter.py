import matplotlib as plt
import numpy as np
import cv2
import os
import math
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

#Thesea re the parameters that we are going to use
accum_res  = 3 # image resolution/accum resolution
min_between = 5 #Min dist between circles. 
minRadius = 600 #Min radius of a circle. 
maxRadius= 680 #The bigest circle expected
Canny_thr = 300 #anything above that is an edge automatically in Canny, the lower threshold is half of that.
Accum_thr = 100 #accumulator threshold for the circle centers at the detection stage

params_Hough = [accum_res, min_between, Canny_thr, Accum_thr, minRadius, maxRadius]

def ScaleImage(image):
    #resize image to ~1500x1500
    Width = image.shape[1]
    Scale = Width/1500
    new_size = (int(image.shape[1]/Scale), int(image.shape[0]/Scale)) 
    img_resized = cv2.resize(image, new_size )
    return img_resized

def FindCircles(params, scaled_img):

    #Convert to grayscale
    img_gr = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

    #Gausian filetr noise
    filtered = cv2.medianBlur(img_gr, 3)
    
    #Enhance contrast
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+107))    ))

    img_contrast = cv2.LUT(filtered, lookUpTable)    
    #Run Hough circle with the params tailored to find big circles.
    circs = cv2.HoughCircles(image = img_contrast, 
                    method = cv2.HOUGH_GRADIENT,
                    dp = params[0],
                    minDist = params[1],
                    param1 = params[2], param2 = params[3],
                    minRadius = params[4], maxRadius = params[5])
    #Round all numbers to the nearst int                
    if circs is not None:
        circs = np.uint16(np.around(circs))
        return circs[0,:], img_contrast
    else:
        return [], img_contrast

def DrawCircles(circle_array, target):

    #Find the median center
    vector_dict = []
    #Calculate the vector length 
    for circ in circle_array:
        vector_dict.append(circ[0]**2+circ[1]**2)
    
    #Find median vector length
    median_len = np.median(vector_dict)

    #Find how far away each vector is from median length
    for circ in vector_dict:
        circ = abs(circ-median_len)
    
    index_min = min(range(len(vector_dict)), key=vector_dict.__getitem__)

    X_mean = circle_array[index_min][0]
    Y_mean = circle_array[index_min][1]

    #Draw circles and their centers
    for i in circle_array:
        '''
        if i[0] == X_mean and i[1] == Y_mean:
            # draw the outer circle
            cv2.circle(target ,(i[0],i[1]),i[2],(0,0,255),10)
            # draw the center of the circle
            cv2.circle(target ,(i[0],i[1]),4,(0,255,0),10)
            radius_txt = str(i[2])
            cv2.putText(target, radius_txt, (i[0],i[1]), cv2.FONT_HERSHEY_PLAIN, 5, (128, 128, 0), 4)
        '''
        # draw the outer circle
        cv2.circle(target ,(i[0],i[1]),i[2],(0,255,0),4)
        # draw the center of the circle
        cv2.circle(target ,(i[0],i[1]),4,(0,0,255),4)

        #Print the radius at the center
        radius_txt = str(i[2])
        cv2.putText(target, radius_txt, (i[0],i[1]), cv2.FONT_HERSHEY_PLAIN, 5, (128, 128, 0), 4)
    #Print number of circles
    num_cir = str(len(circle_array))
    cv2.putText(target, num_cir, (50,50), cv2.FONT_HERSHEY_PLAIN, 20 (150, 150, 0), 5)
        

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
    # I modified this code from https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
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
    '''
    # Find homography
    if len(points1) >=4 and len(points2)>=4:
        h, mask_1 = cv2.findHomography(points1, points2, cv2.RANSAC,ransacReprojThreshold=5.0)
    else:
        return im1, []
    '''
    # Calculate the transformation matrix using cv2.getAffineTransform()
    points_source = points1[0:3]
    points_ref = points2[0:3]
    h= cv2.getAffineTransform(points_source, points_ref)
    
    if h is not None:
        '''
        # Use homography for Perspective transform
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (height, width), np.array([]))
        '''
        # Use transformation matrix for affine transform
        height, width, channels = im2.shape
        im1Reg = cv2.warpAffine(im1, h, (height, width), np.array([]))
        

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

    #Scale Image
    img_scaled = ScaleImage(img)
    

    #Take scaled image, find circles and return a list of circles
    Circles, leveled = FindCircles(params_Hough, img_scaled)


    #Draw the circles on the image provided
    if len(Circles)>0:
        DrawCircles(Circles, img_scaled)
	
    #Write the image with circles
    cv2.imwrite(processed_path+'/'+file, img_scaled)
    #Write the pre-processed image
    cv2.imwrite(processed_path+'/'+file+'_proc', leveled)

