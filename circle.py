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
    dist_from_center_sq = (Y - center[1])**2 + (X-center[0])**2
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
    Intensity_f_R = [   Intensity(image, center, i, step)    for i in range(min_radius, max_radius, step) ]
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
    #Take blue channel
    img_gr = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,12)
    img_gr = cv2.bitwise_not(img_gr)

    #Lets expand array of circles to store additional infor we find out
    #[0] - x, [1] - y, [2] - radius, [3] - MaxRadius, [4] - MaxDeriv
    zeros = np.zeros((circles.shape[0], 3), dtype = int)
    circles = np.concatenate(  (circles, zeros),  axis = 1)  

    for i, circle in enumerate(circles):
        #Find the max radius that can possibly be at this cneter point
        #For this find how far away this point is from the center of the image
        #Keep in mind that image.shape[height, width], but circle[x,y,r]
        Y_offset = abs(image.shape[0]/2-circle[0])
        X_offset = abs(image.shape[1]/2-circle[1])
        max_X = image.shape[1]//2-X_offset
        max_Y = image.shape[0]//2-Y_offset
        max_R = min(max_X, max_Y)
        circle[3]=max_R
        if max_R > 510:
            #Calculate how intensity changes (i.e. derivative) as the radius increases.       
            Deriv_Step = 10

            Deriv_f_R = Deriv(img_gr,  (circle[0], circle[1]), 500, 720, Deriv_Step)
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
    Fine_Intensity = [   Intensity(img_gr, (BestCircle[0], BestCircle[1]), i, FineStep)    for i in range(500, BestCircle[3], FineStep) ]
    Deriv_Brightest = Deriv(img_gr,  (BestCircle[0], BestCircle[1]), 500, BestCircle[3], FineStep )
    Fine_Int = np.round(Fine_Intensity,2)
    Fine_Deriv = np.round(Deriv_Brightest, 2)
    # print(' ')
    # print(Fine_Int)
    # print(Fine_Deriv)

    # Start forming the list we will return: add center coordinate
    return_value = [BestCircle[0], BestCircle[1]]


    # Go through the deriv and if the point and the next point is above 1500, say that is where ROI stops
    # Don't forget that we only calculate starting from 500 pixels from the center, so add those 500px
    for i in range(len(Fine_Int)-1):
        Int_crossed = False
        Deriv_crossed = False
        if Fine_Int[i] > 0.09:
            Int_crossed = True
        if Fine_Deriv[i] > 0.09:
            Deriv_crossed = True
        if Deriv_crossed | Int_crossed:
            return_value.append(500+i*FineStep)
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