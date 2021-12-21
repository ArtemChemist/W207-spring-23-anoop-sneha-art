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
onlyfiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
'''
#Read background image
bckg = cv2.imread(folder_path+'/bckg.jpg',cv2.IMREAD_GRAYSCALE)

#Create mask for a circle in the center of the image area
H, W = bckg.shape
mask = np.full((H,W), 0, dtype=np.uint8)
radius = W/4
for i in range(H):
    for k in range(W):
        Is_In_ROI = (i-H/2)**2 + (k-W/2)**2 < radius**2
        mask[i,k] = Is_In_ROI

#Apply mask 
res = bckg[mask]

#Find the threshold as 5x the average of the background
thr = 5*np.median(res)
'''
T1 = 60
T2 = 100
for file in onlyfiles:
    #Read the image images
    img = cv2.imread(folder_path+'/'+file)

    #Conver to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Filter out dust and pepper
    img_blur = cv2.medianBlur(img_gray,17)

    H, W = img_blur.shape
    radius_sq = (0.34*W)**2
    for i in range(1,H-1):
        for k in range(1,W-1):
            #Crop out the LEDs
            if (i-H/2)**2 + (k-W/2)**2 > radius_sq:
                img_blur[i,k] = 0

            #Suppress low pixels
            if img_blur[i,k] < T1:
                img_blur[i,k] = 0

            #Enhance high pixels
            elif img_blur[i,k] > T2:
                img[i,k,1] = 255
                img_blur[i,k] = 255
            
            #Check surrounding if in doubt
            else:
                if (   img_blur[i-1,k-1] > T2 or
                       img_blur[i,k-1] > T2 or
                       img_blur[i-1,k] > T2 or
                       img_blur[i+1,k+1] > T2 or
                       img_blur[i+1,k] > T2 or
                       img_blur[i,k+1] > T2 or
                       img_blur[i-1,k+1] > T2 or
                       img_blur[i+1,k-1] > T2
                    ):
                    img[i,k,1] = 255
                    img_blur[i,k] = 255

                else:
                    img_blur[i,k] = 0

                
    #thr_image = img_gray[:,:]<50
    #img_gray[thr_image] = 0

    #Apply dynamic thresholding
    #niBlack = cv2.ximgproc.niBlackThreshold(img_blur, 255, cv2.THRESH_BINARY, 101,0.7,cv2.ximgproc.BINARIZATION_NIBLACK)
    #Write the final image

    cv2.imwrite(processed_path+'/'+file, img)