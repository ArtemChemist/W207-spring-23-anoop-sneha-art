from cmath import sin
import numpy as np
import cv2
import os
import math
from os import listdir
from os.path import isfile, join
#from skimage.filters import threshold_sauvola
import time
from aux import Circular_mask, ScaleImage, DrawCircles, EnhanceContrast
from circle import Find_Best_Circle, Find_Optimum_Circles



def main():
    #get a list of files in the folder with pics
    folder_path = os.path.dirname(__file__)+'/Smpl_Im'
    processed_path = os.path.dirname(__file__)+'/Smpl_Thresh'
    file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    start_time = time.time()
    for file in file_names:
        begin_time = time.time()
        new_name = os.path.join(processed_path, file)
        try:
            #Read the image
            img = cv2.imread(folder_path+'/'+file)
            print(f"Processing {file[0:-4]}", end = "->  ")

            # Scale all images to the same 1512 pixels
            # Keep resolution fairly high for Hough transform
            img_scaled = ScaleImage(img, width=1536)

            #Hough Transform parameters
            accum_res  = 3 # image resolution/accum resolution, 4 means accum is 1/4th of image
            min_between = 4 #Min dist between circles. 
            minRadius = 610 #Min radius of a circle. 
            maxRadius= 740 #The bigest circle expected
            Canny_thr = 1100 #anything above that is an edge automatically in Canny, the lower threshold is half of that.
            Accum_thr = 1100 #accumulator threshold for the circle centers at the detection stage
            params_Hough = [accum_res, min_between, Canny_thr, Accum_thr, minRadius, maxRadius]

            Circles = Find_Optimum_Circles(params_Hough, img_scaled, 10, 30)
            print(f"{len(Circles):10} circles")

            #If there are any circles, filter them by centricity and dI/dR
            if len(Circles)>0:
                # In the list of centers, find the center of the true ROI
                # ROI is the circle with the sharpest brigtnest change
                # Brightness = intensity of a ring
                # Brightness change = its derivative on radius
                
                BestCirc = Find_Best_Circle(Circles, img_scaled)
                print(f" found best in {(time.time() - begin_time):.2f} sec", end = " - ")

                # Draw the best circel on the image
                # img_scaled= DrawCircles(BestCirc, img_scaled)

                if len(BestCirc[0])>2:
                    #Create a new image that is a square bounding this circular ROI
                    x = BestCirc[0][0]  # X coordinate of center
                    y = BestCirc[0][1]  # Y coordinate of center
                    r = BestCirc[0][2]  # Radius
                    cut_image = img_scaled[y-r:r+y, x-r:x+r, :  ]

                    #Apply circluar mask,set everything else to 0.
                    mask = Circular_mask(cut_image.shape[0], cut_image.shape[1], radius = r)
                    cut_image[~mask] = 0

                    # Enhance contrast
                    contrast_enh = EnhanceContrast(cut_image, -20, 45)


                    #Threshold the edges to remove residual LED glare
                    th = cv2.adaptiveThreshold(contrast_enh[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,41,22)
                    th = cv2.bitwise_not(th)
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(th , kernel, iterations = 10)
                    edge_mask = Circular_mask(dilated.shape[0], dilated.shape[1], radius = int(0.48*dilated.shape[0]))
                    mask = (~edge_mask) & (dilated > 244)
                    contrast_enh[mask] = 0


                    #Write final file to disk
                    #Scale the new image to 1024 pixels
                    # We do not need more than 512 for NN anyway
                    final = ScaleImage(contrast_enh, width = 1024)
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    cv2.imwrite(new_name , final)
                    print(f" done in {(time.time() - begin_time):.2f} sec")
        except:
            print(f'Something went wrong on {file}')
    print(f"Overall for {len(file_names)} images {time.time()-start_time:.1f}sec")

if __name__ == '__main__':
    main()

