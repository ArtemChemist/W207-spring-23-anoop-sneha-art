import numpy as np
import cv2
import math


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
    dist_from_center_sq = (Y - center[1])**2 + (X-center[0])**2 #We get x,y coord of center, NOT y,x
    mask = (dist_from_center_sq >= in_radius_sq ) & (dist_from_center_sq <= out_radius_sq)
    my_mask = np.asarray(mask, dtype="uint8")
    hist = cv2.calcHist([image], [0], my_mask, [128], [0,256])
    fraction = hist[-1]/sum(hist)

    return fraction[0]

def Deriv(image= np.array, center = (int, int),  min_radius = int, max_radius = int, step = int):
    '''
    Finds derivative of the d(circumference integral)/d(distanse from center)
    '''
    #First Calucalte how integral brightness of the circles dpends on their radius
    #Do that starting from fairly large radius, to start close to the edge already
    h = image.shape[0] # Image height
    w = image.shape[1] # Image width
    Y, X = np.ogrid[:h, :w]
    dist_from_center_sq = (Y - center[1])**2 + (X-center[0])**2 #We get x,y coord of center, NOT y,x
    Intensity_f_R = []
    
    for i in range(min_radius, max_radius, step):
        in_radius_sq = i**2
        out_radius_sq = (i+step)**2
        my_mask = np.asarray((dist_from_center_sq >= in_radius_sq ) & (dist_from_center_sq <= out_radius_sq), dtype="uint8")
        hist = cv2.calcHist([image], [0], my_mask, [128], [0,256])
        Intensity_f_R.append(hist[-1]/sum(hist))

    Intensity_f_R = np.array(Intensity_f_R)
    # Now calculate derivative of this function
    i_next = np.arange(1, len(Intensity_f_R))
    i = np.arange(0, len(Intensity_f_R)-1)
    to_return = np.add(Intensity_f_R[i_next],-Intensity_f_R[i])
    #return this reriative
    return to_return


def Find_All_Circles(params, input_img):

    #Convert to grayscale
    img_gr = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    #img_gr  = input_img[:,:,0]

    #img_gr = cv2.adaptiveThreshold(input_img[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,151,12)
    #img_gr = cv2.bitwise_not(img_gr)


    # blur = cv2.GaussianBlur(input_img[:,:,0],(5,5),0)
    # ret3,img_gr = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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

def Find_Best_Circle(circles, image):
    '''
    Find the circle that had the sharpest change brightness as we move away from the center
    '''
    #Take blue channel, threshold the hell out of it, to leave only LED
    img_gr = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,12)
    img_gr = cv2.bitwise_not(img_gr)

    #Lets expand array of circles to store additional infor we find out
    #[0] - x, [1] - y, [2] - radius, [3], [4] - MaxDeriv, [5] - Rad for max deriv,
    zeros = np.zeros((circles.shape[0], 3), dtype = int)
    circles = np.concatenate(  (circles, zeros),  axis = 1)

    Deriv_Step = 10
    Start_rad = 200
    for i, circle in enumerate(circles):
        #Calculate how intensity changes (i.e. derivative) as the radius increases.       
        Deriv_f_R = Deriv(img_gr,  (circle[0], circle[1]), Start_rad, 800, Deriv_Step)

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
    # Get full Deriv and Intensity for this one circle
    #  
    FineStep = 2
    best_center = (BestCircle[0], BestCircle[1])
    Fine_Intensity = [   Intensity(img_gr, best_center , i, FineStep)    for i in range(Start_rad, 750, FineStep) ]
    Deriv_Brightest = Deriv(img_gr, best_center, Start_rad, 750, FineStep )
    Fine_Int = np.round(Fine_Intensity,2)
    Fine_Deriv = np.round(Deriv_Brightest, 2)
    print(' ')
    #print(Fine_Int)
    #print(Fine_Deriv)

    # Start forming the list we will return: add center coordinate
    return_value = [BestCircle[0], BestCircle[1]]

    # Go through the deriv and if it crosse some pre-defined trhreshold
    # Add Start_rad pixels from the center, we only started there, not at 0
    for i in range(len(Fine_Int)-1):
        Int_crossed = False
        Deriv_crossed = False
        if Fine_Int[i] > 0.09:
            Int_crossed = True
        if Fine_Deriv[i] > 0.09:
            Deriv_crossed = True
        if Deriv_crossed | Int_crossed:
            best_rad = Start_rad+i*FineStep
            print(f"Crossed at {best_rad }")
            return_value.append(best_rad )
            break

    
    return [return_value]

def Find_Optimum_Circles(params_Hough, img, min_num, max_num):

    # Take scaled image, find circles and return an array of circles
    # Run quick optimization to get reasonable number of circles
    for i in range (0, 1100, 10):
        params_Hough[2] = 1100 - i
        params_Hough[3] = 1100 - i
        Circles, _ = Find_All_Circles(params_Hough, img)
        # print(f"{i}->{len(Circles)}", end = ' - ')
        if len(Circles) > min_num:
            if len(Circles) <max_num:
                break
            else:
                for k in range (10):
                    params_Hough[2] = 1100-i + k
                    params_Hough[3] = 1100-i + k
                    # print(f"{k}->{len(Circles)}", end = ' - ')
                    Circles, _ = Find_All_Circles(params_Hough, img)
                    if len(Circles) <max_num:
                        break
            break
    return Circles