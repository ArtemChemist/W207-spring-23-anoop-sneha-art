import numpy as np
import cv2
import math
from scipy import signal


def Intensity(image = np.array, center = (int,int), radius = int, band_width = int):
    '''
    Returns fraction of pixels bighter then 150 in the band 
    '''
    # Make a circular band mask
    in_radius_sq = radius**2
    out_radius_sq = (radius+band_width)**2
    h = image.shape[0] # Image height
    w = image.shape[1] # Image width
    Y, X = np.ogrid[:h, :w]
    dist_mtx = (Y - center[1])**2 + (X-center[0])**2 #We get x,y coord of center, NOT y,x
    my_mask = np.asarray((dist_mtx >= in_radius_sq ) & (dist_mtx <= out_radius_sq), dtype="uint8")
    hist = cv2.calcHist([image], [0], my_mask, [128], [0,256])
    fraction = hist[-1]/sum(hist)

    return fraction[0]

def Deriv(image= np.array, center = (int, int),  min_radius = int, max_radius = int, step = int):
    '''
    Finds derivative of the brightness of ring over its radius
    '''
    # First Calucalte brightness of the ring as a function of its radius
    # I define brightness of the ring as the fraction of the pixels with the highest brighness
    # Use formula of a circle to make an array of 
    # how far each in the image is from the center of the given ring pixel is 
    h = image.shape[0] # Image height
    w = image.shape[1] # Image width
    Y, X = np.ogrid[:h, :w]
    dist_mtx = (Y - center[1])**2 + (X-center[0])**2 #We get x,y coord of center, NOT y,x
    
    bright_arr = [] # list that holds bightness for each radius
    for i in range(min_radius, max_radius, step):
        # Make a mask shape of the ring
        in_radius_sq = i**2
        out_radius_sq = (i+step)**2
        my_mask = np.asarray((dist_mtx >= in_radius_sq ) & (dist_mtx <= out_radius_sq), dtype="uint8")
        # Calculate histogram of the masked image
        # Note, cv2 does that much faster then straight numpy.
        # This appears to be the speed limiting step
        hist = cv2.calcHist([image], [0], my_mask, [128], [0,256])
        bright_arr.append((hist[-1]/sum(hist))[0])
    bright_arr = np.array(bright_arr)

    # Now calculate derivative of this function
    i_next = np.arange(1, len(bright_arr))
    i = np.arange(0, len(bright_arr)-1)
    deriv = np.add(bright_arr[i_next],-bright_arr[i])

    return deriv


def Find_All_Circles(params, input_img):
    """
    Runs Hough transform and returns an array of circles
    """
    #Convert to grayscale
    img_gr = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    #Enhance contrast
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+120))    ))
    img_gr = cv2.LUT(img_gr, lookUpTable)

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
        return circs[0,:], img_gr
    else:
        return [], img_gr

def Find_Best_Center(circles, image, start: int, end: int, step:int):
    '''
    Find the circle that had the sharpest change brightness as we move away from its center
    '''
    Deriv_Step = step
    Start_rad = start
    End_rad = end
    #Take blue channel, threshold the hell out of it, to leave only LED
    img_gr = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,81,15)
    img_gr = cv2.bitwise_not(img_gr)
    kernel = np.ones((2, 2), np.uint8)
    img_gr = cv2.erode(img_gr , kernel, iterations = 3)

    #Lets expand array of circles to store additional infor we find out
    #[0] - x, [1] - y, [2] - radius, [3], [4] - MaxDeriv, [5] - Rad for max deriv,
    zeros = np.zeros((circles.shape[0], 3), dtype = int)
    circles = np.concatenate(  (circles, zeros),  axis = 1)

    for i, circle in enumerate(circles):
        #Calculate how intensity changes (i.e. derivative) as the radius increases.       
        Deriv_f_R = Deriv(img_gr,  (circle[0], circle[1]), Start_rad, End_rad, Deriv_Step)
        # Do median filtration
        Deriv_f_R  = signal.medfilt(Deriv_f_R , 3)
        #Find what was the sharpest change for this circle, 
        MaxDeriv = np.max(Deriv_f_R)
        circle[4] =  int(MaxDeriv * 1000) # we strore it in int array

        #Find where this sharp change occured
        R_of_Max = Start_rad + (Deriv_f_R.argmax()+1)*Deriv_Step
        circle[5] = R_of_Max

    #Find the circle that had the most abrupt change.
    #The idea is that as we go out from the most central point will have all LEDs come into view at once
    #As opposed to point that is off-center, where expanding circle wil hit only few LEDs at a time.
    #For that, sort all circles by the maxDrevi Value
    sorted_circles = circles[np.argsort(circles[:, 4])]
    BestCircle = sorted_circles[-1]

    # Now that we know where is the true center of the ROI, let's find its true radius.
    # Get full Deriv for this one circle
    FineStep = 2
    best_center = (BestCircle[0], BestCircle[1])

    Deriv_Brightest = Deriv(img_gr, best_center, Start_rad, End_rad, FineStep )
    Deriv_Brightest[Deriv_Brightest<0]=0
    Fine_Deriv  = signal.medfilt(Deriv_Brightest, 3)
    Fine_Deriv = np.round(Fine_Deriv, 3)
    #print(Fine_Deriv)

    # Start forming the list we will return: add center coordinate
    return_value = [BestCircle[0], BestCircle[1], BestCircle[5]]

    # Go through the deriv until it crosses the pre-defined trhreshold
    # Add Start_rad pixels from the center, we only started there, not at 0
    for i in range(len(Fine_Deriv)):
        Deriv_crossed = False
        if Fine_Deriv[i] > 0.035:
            Deriv_crossed = True
        if Deriv_crossed:
            best_rad = Start_rad+i*FineStep
            print(f"First iter rad {best_rad }")
            return_value[2] = best_rad
            break
    return [return_value]

def Find_Best_Radius(center, image, start: int, end: int, step:int, thresh):
    '''
    Find the circle that had the sharpest change brightness as we move away from its center
    '''
    #Take blue channel, threshold the hell out of it, to leave only LED
    img_gr = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,81,15)
    img_gr = cv2.bitwise_not(img_gr)
    # kernel = np.ones((2, 2), np.uint8)
    # img_gr = cv2.erode(img_gr , kernel, iterations = 3)

    Int_arr = [Intensity(img_gr, center, rad, step) for rad in range (start, end, step)]
    Int_arr = signal.medfilt(Int_arr, 3)
    Int_arr  = np.round(Int_arr , 3)
    # print(Int_arr )

    # Go through the deriv until it crosses the pre-defined trhreshold
    # Add Start_rad pixels from the center, we only started there, not at 0
    for i in range(len(Int_arr )-1, 0, -1):
        if Int_arr [i] < thresh:
            best_rad = start+(i+1)*step
            print(f"Second iter rad {best_rad }")
            return( best_rad )
    return end

def Find_Optimum_Circles(params_Hough, img, min_num, max_num):
    """
    Takes scaled image, finds circles and return an array of circles
    Runs quick optimization of Hough trnasform params
    to get reasonable number of circles
    """
    to_return = []
    for i in range (0, 1100, 10):
        params_Hough[2] = 1100 - i
        params_Hough[3] = 1100 - i
        Circles, _ = Find_All_Circles(params_Hough, img)
        #print(f"{i}->{len(Circles)}", end = ' - ')
        if len(Circles) > min_num:
            tmp = len(Circles)
            if len(Circles) <max_num:
                to_return = Circles
                break
            else:
                for k in range (10):
                    params_Hough[2] = 1100-i + k
                    params_Hough[3] = 1100-i + k
                    #print(f"{k}->{len(Circles)}", end = ' - ')
                    Circles, _ = Find_All_Circles(params_Hough, img)
                    if (len(Circles) <max_num) | (len(Circles) > tmp):
                        break
            to_return = Circles
            break
    return to_return