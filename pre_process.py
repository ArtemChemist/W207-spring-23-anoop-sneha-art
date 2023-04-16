from cmath import sin
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join
#from skimage.filters import threshold_sauvola
import time
from aux import Circular_mask, ScaleImage, DrawCircles, EnhanceContrast
from circle import Find_Best_Center, Find_Optimum_Circles, Find_Best_Radius



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
                # For that: Check along the radius, from start to end, steping every step pixels
                # These are different form similar numbers in Hough params
                # Becasue Hough needs to be stable, but this procidure needs to be precise
                BestCirc = Find_Best_Center(circles = Circles,
                                                image= img_scaled,
                                                start= 480,
                                                end = 720,
                                                step = 10)

                # Draw the best circel on the image
                # img_scaled= DrawCircles(BestCirc, img_scaled)

                if len(BestCirc[0])>2:
                    #Create a new image that is a square bounding this circular ROI
                    x = BestCirc[0][0]  # X coordinate of center
                    y = BestCirc[0][1]  # Y coordinate of center
                    r = BestCirc[0][2]  # Radius

                    # Apply circluar mask around best circle, set everything outside to 0
                    h = img_scaled.shape[0]
                    w = img_scaled.shape[1]
                    hole = Circular_mask(h, w, center = (x,y), radius = r)
                    img_scaled[~hole] = 0
                    top_edge = max(y-r, 0)
                    bottom_edge = min(y+r, h)
                    left_edge = max(x-r,0)
                    right_edge = min(x+r,w)
                    #Cut out the region of interest around this circle
                    First_Iter = img_scaled[top_edge:bottom_edge, left_edge:right_edge, :]
                    # Stardardize all images to the same size
                    # That makes next step more stable
                    First_Iter = ScaleImage(First_Iter, width = 1024)

                    # Enhance contrast
                    First_Iter = EnhanceContrast(First_Iter, -20, 45) 

                    # Second iteration - now go by intensity, not derivative
                    # Also, go from outside in
                    h2 = First_Iter.shape[0]
                    w2 = First_Iter.shape[1]
                    y2 = First_Iter.shape[0]//2
                    x2 = First_Iter.shape[1]//2
                    fine_step = 10
                    end =  x2

                    r2 = Find_Best_Radius(center = (x2,y2), 
                                            image = First_Iter,
                                            start= end-10*fine_step, end = end, step = fine_step,
                                            thresh= 0.025)

                    new_hole = Circular_mask(h2, w2, center = (x2,y2), radius = r2)
                    First_Iter[~new_hole] = 0
                    top_edge = max(y2-r2, 0)
                    bottom_edge = min(y2+r2, h2)
                    left_edge = max(x2-r2,0)
                    right_edge = min(x2+r2,w2)
                    ROI_image = First_Iter[top_edge:bottom_edge, left_edge:right_edge, :]
                    
                    """
                    #Threshold the edges to remove residual LED glare
                    th = cv2.adaptiveThreshold(contrast_enh[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,41,22)
                    th = cv2.bitwise_not(th)
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(th , kernel, iterations = 10)
                    edge_mask = Circular_mask(dilated.shape[0], dilated.shape[1], radius = int(0.48*dilated.shape[0]))
                    mask = (~edge_mask) & (dilated > 244)
                    contrast_enh[mask] = 0
                    """

                    # Write final file to disk
                    # Scale the new image to 1024 pixels
                    # We do not need more than 512 for NN anyway
                    final = ScaleImage(ROI_image, width = 1024)
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    cv2.imwrite(new_name , final)
                    print(f" done in {(time.time() - begin_time):.2f} sec")
        except:
            print(f'Something went wrong on {file}')
    print(f"Overall for {len(file_names)} images {time.time()-start_time:.1f}sec")

if __name__ == '__main__':
    main()

