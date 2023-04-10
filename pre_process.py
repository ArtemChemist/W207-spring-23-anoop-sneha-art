from cmath import sin
import numpy as np
import cv2
import os
import math
from os import listdir
from os.path import isfile, join
#from skimage.filters import threshold_sauvola
import time

def create_circular_mask(h, w, center=None, radius=None):
    '''
    Creates a mask of dimentions, height = h, width = x, with a circle marked true, 
    located at the center and having radius  = radius
    I modified it from here:
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''
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

def ScaleImage (image, width = 1512, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def EnhanceContrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def FindCircles(params, input_img):

    #Convert to grayscale
    #img_gr = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    img_gr  = input_img[:,:,0]

    #Enhance contrast
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+120))    ))

    img_contrast = cv2.LUT(img_gr, lookUpTable)    
    #Run Hough circle with the params tailored to find big circles.
    circs = cv2.HoughCircles(image = img_gr, 
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
    '''
    Draws green circles with red centers on the "target" image. Assumes an array of circles "circle_array"
     to have center_x at indx 0, center y ant index 1 and radius at index 2.
    If some circles do not have radius defined, draws them as red circls radius 500
    '''

    #Draw circles and their centers
    for i in circle_array:

        if len(i) ==2:
            # draw the center of the circle
            cv2.circle(target ,(i[0],i[1]),4,(0,0,255),4)
            # Draw the outer circle in red
            cv2.circle(target ,(i[0],i[1]),500,(0,0,255),8)


        if len(i) ==3:
            # draw the center of the circle
            cv2.circle(target ,(i[0],i[1]),4,(0,0,255),4)
            # draw the outer circle
            cv2.circle(target ,(i[0],i[1]),i[2],(0,255,0),4)
            #Print the radius at the center
            radius_txt = str(i[2])
            cv2.putText(target, radius_txt, (i[0],i[1]), cv2.FONT_HERSHEY_PLAIN, 5, (128, 128, 0), 4)
    
    #Print number of circles
    num_cir = str(len(circle_array))
    cv2.putText(target, num_cir, (100,200), cv2.FONT_HERSHEY_PLAIN, 8, (150, 150, 0), 12)
        
def Circ_Integral(image = np.array, center = (int,int), radius = int, band_width = int):
    '''
    Returns fraction of pixels bighter then 150 in the band 
    '''
    # Make a circular band mask
    in_radius_sq = radius**2
    out_radius_sq = (radius+band_width)**2
    h = image.shape[0] # Image height
    w = image.shape[1] # Image width
    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (Y - center[0])**2 + (X-center[1])**2
    mask = (dist_from_center_sq >= in_radius_sq ) & (dist_from_center_sq <= out_radius_sq)
    my_mask = np.asarray(mask, dtype="uint8")
    hist = cv2.calcHist([image], [0], my_mask, [16], [0,256])
    fraction = hist[-1]/sum(hist)

    return fraction[0]

def Deriv_Intensity_f_R(image= np.array, center = (int, int),  min_radius = int, max_radius = int, step = int):
    '''
    Finds derivative of the d(circumference integral)/d(distanse from center)
    '''
    #First Calucalte how integral brightness of the circles dpends on their radius
    #Do that starting from fairly large radius, to start close to the edge already
    Intensity_f_R = [   Circ_Integral(image, center, i, step)    for i in range(min_radius, max_radius, step) ]
    Intensity_f_R = np.array(Intensity_f_R)
    # Now calculate derivative of this function
    i_next = np.arange(1, len(Intensity_f_R))
    i = np.arange(0, len(Intensity_f_R)-1)
    Deriv = np.add(Intensity_f_R[i_next],-Intensity_f_R[i])
    #return this reriative
    return Deriv

def FindBestCircle(circles, image):
    '''
    Find the circle that had the sharpest change brightness as we move away from the center
    '''
    #Take blue channel
    img_gr = image[:,:,0]
    
    #Enhance contrast, but not as dramatically as for Hugh transform
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+120))    ))

    img_contrast = cv2.LUT(img_gr, lookUpTable)    

    #Lets expand array of circles to store additional infor we find out
    #[0] - x, [1] - y, [2] - radius, [3] - MaxRadius, [4] - MaxDeriv
    zeros = np.zeros((circles.shape[0], 3), dtype = int)
    circles = np.concatenate(  (circles, zeros),  axis = 1)  

    for i, circle in enumerate(circles):
        #Find the max radius that can possibly be at this cneter point
        #For this find how far away this point is from the center of the image
        #Keep in mind that image.shape[height, width], but circle[x,y,r]
        Y_offset = abs(image.shape[0]/2-circle[1])
        X_offset = abs(image.shape[1]/2-circle[0])
        max_X = int(image.shape[1]/2-X_offset)
        max_Y = int(image.shape[0]/2-Y_offset)
        max_R = min(max_X, max_Y)
        circle[3]=max_R
        if max_R > 510:
            #Calculate how intensity changes (i.e. derivative) as the radius increases.       
            Deriv_Step = 10

            Deriv_f_R = Deriv_Intensity_f_R(img_contrast,  (circle[0], circle[1]), 500, max_R, Deriv_Step)
            #Find what was the sharpest change for this circle, 
            MaxDeriv = np.max(Deriv_f_R)
            circle[4] =  int(MaxDeriv * 1000) # we strore it in int array
            #Find where this sharp change occured
            R_of_Max = 500 + (Deriv_f_R.argmax()+1)*Deriv_Step
            circle[5] = R_of_Max

    #Find the circle that had the most abrupt change.
    #The idea is that as we go out from the most central point will have all LEDs come into view at once
    #As opposed to point that is off-center, where expanding circle wil hit only few LEDs at a time.
    #For that, sort all circles by the maxDrevi Value
    sorted_circles = circles[np.argsort(circles[:, 4])]
    BestCircle = sorted_circles[-1]

    # Now that we know where is the true center of the ROI, let's find its true radius.
    # Get full Deriv for this one circle, all the way to the max possible radius
    #  
    FineStep = 2
    Fr_Brightness = [   Circ_Integral(img_contrast, (BestCircle[0], BestCircle[1]), i, FineStep)    for i in range(500, BestCircle[3], FineStep) ]
    Deriv_Brightest = Deriv_Intensity_f_R(img_contrast,  (BestCircle[0], BestCircle[1]), 500, BestCircle[3], FineStep )
    Deriv_Brightest[Deriv_Brightest<0] = 0
    Fr = np.round(Fr_Brightness/np.max(Fr_Brightness),2)
    Deriv_Brightest = Deriv_Brightest/np.max(Deriv_Brightest)
    Deriv = np.round(Deriv_Brightest, 2)
    #print(Fr)
    #print(Deriv)

    # Start forming the list we will return: add center coordinate
    return_value = [BestCircle[0], BestCircle[1]]


    # Go through the deriv and if the point and the next point is above 1500, say that is where ROI stops
    # Don't forget that we only calculate starting from 500 pixels from the center, so add those 500px
    for i in range(len(Fr)-1):
        Int_crossed = False
        Deriv_crossed = False
        if Fr[i] > 0.14:
            Int_crossed = True
        if Deriv[i] > 0.14:
            Deriv_crossed = True
        if Int_crossed | Deriv_crossed:
            return_value.append(500+i*FineStep)
            break

    
    return [return_value]


def main():
    #get a list of files in the folder with pics
    folder_path = os.path.dirname(__file__)+'/Smpl_Im'
    processed_path = os.path.dirname(__file__)+'/Smpl_Thresh'
    file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in file_names:
        new_name = os.path.join(processed_path, file)
        try:
            #Read the image
            img = cv2.imread(folder_path+'/'+file)
            print(f"Processing {file[0:-4]}", end = "->  ")

            #Scale Image
            img_scaled = ScaleImage(img)

            #These are the Hough Transform parameters that we are going to use
            accum_res  = 3 # image resolution/accum resolution, 4 means accum is 1/4th of image
            min_between = 4 #Min dist between circles. 
            minRadius = 600 #Min radius of a circle. 
            maxRadius= 740 #The bigest circle expected
            Canny_thr = 1100 #anything above that is an edge automatically in Canny, the lower threshold is half of that.
            Accum_thr = 1100 #accumulator threshold for the circle centers at the detection stage
            params_Hough = [accum_res, min_between, Canny_thr, Accum_thr, minRadius, maxRadius]



            # Take scaled image, find circles and return an array of circles
            # Run quick optimization to get reasonable number of circles
            start_time = time.time()
            for i in range (0, 1100, 25):
                params_Hough[2] = 1100 - i
                params_Hough[3] = 1100 - i
                #print(i, end = ' - ')
                Circles, _ = FindCircles(params_Hough, img_scaled)
                if len(Circles) > 20:
                    if len(Circles) <50:
                        break
                    else:
                        for k in range (25):
                            params_Hough[2] = 1100-i + k
                            params_Hough[3] = 1100-i + k
                            #print(k, end = ' - ')
                            Circles, _ = FindCircles(params_Hough, img_scaled)
                            if len(Circles) <50:
                                break
                    break

            print(f"   {len(Circles):10} circles, in {(time.time() - start_time):.2f} sec", end = " ->")

            #If there are any circles, filter them by centricity and dI/dR
            if len(Circles)>0:
                # In the list of centers, find the center of the true ROI
                # ROI is the circle with the sharpest brigtnest change
                # Brightness = integral of intensity along circumference
                # Brightness change = its derivative on radius
                # I.e. df/dr, where f = integral(intesity)*d(circ) 
                
                begin_time = time.time()
                BestCirc = FindBestCircle(Circles, img_scaled)
                print(f" found best in {(time.time() - begin_time):.2f} sec")

                # Draw the best circel on the image
                DrawCircles(BestCirc, img_scaled)

                if len(BestCirc[0])>2:
                    #Create a new image that is a square bounding this circular ROI
                    x = BestCirc[0][0]  # X coordinate of center
                    y = BestCirc[0][1]  # Y coordinate of center
                    r = BestCirc[0][2]  # Radius
                    cut_image = img_scaled[y-r:r+y, x-r:x+r, :  ]

                    #Apply circluar mask,set everything else to 0.
                    mask = create_circular_mask(cut_image.shape[0], cut_image.shape[1], radius = r)
                    cut_image[~mask] = 0

                    #Scale the new image to 1500 pixels
                    ROI_img = ScaleImage(cut_image, width = 1024)

                    """#  Enhance contrast
                    contrast_enh = EnhanceContrast(ROI_img, -20, 45)
                    thresh_sauvola = threshold_sauvola(contrast_enh[:,:,2], window_size=20)
                    final = contrast_enh> thresh_sauvola"""
                    #Write final file to disk
                    final = ROI_img
                    if os.path.exists(new_name):
                        os.remove(new_name)
                    cv2.imwrite(new_name , final)
            
            #Write the image with circles
            #cv2.imwrite(processed_path+'/'+file, img_scaled)
        except:
            print(f'Something went wrong on {file}')

if __name__ == '__main__':
    main()

