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

#Everything below T1 is background, everything above T2 is colony
T1 = 60
T2 = 100

for file in file_names:
    #Read the image images
    img = cv2.imread(folder_path+'/'+file)

    #Conver to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Filter out dust and pepper
    img_blur = cv2.medianBlur(img_gray,17)

    #Define the center and the radius of the ROI
    H, W = img_blur.shape
    radius= 0.34*W

    #Set everything outside of the ROI to 0
    mask = create_circular_mask(H, W, (H/2, W/2), radius)
    img_blur[~mask] = 0

    #Set everything below T1 to 0
    img_blur[img_blur<T1] = 0

    #Set everything at or above T2 to 0
    img_blur[img_blur>=T2] = 255

    #Do the hysteresis threshold
    for i in range(1,H-1):
        for k in range(1,W-1):
            neigbours = np.array([
                         img_blur[i-1,k-1], 
                         img_blur[i,k-1],
                         img_blur[i-1,k],
                         img_blur[i+1,k+1],
                         img_blur[i+1,k],
                         img_blur[i,k+1],
                         img_blur[i-1,k+1],
                         img_blur[i+1,k-1]])
            #If no neigbour pixel is positive, set this to negative
            if (img_blur[i,k]<T2 and neigbours.max() <T2):
                img_blur[i,k] = 0
            
            #If there is a positive neigbour, set this to positive
            elif (img_blur[i,k]>T1 and neigbours.max() >=T2):
                img_blur[i,k] = 255

    #Use img_blur as a mask to highlight colonies on teh original image
    img[img_blur>0] = [0,255,0]

    '''            
    #Apply dynamic thresholding
    niBlack = cv2.ximgproc.niBlackThreshold(img_blur, 255, cv2.THRESH_BINARY, 101,0.7,cv2.ximgproc.BINARIZATION_NIBLACK)
    '''

    #Write the final image
    cv2.imwrite(processed_path+'/'+file, img)