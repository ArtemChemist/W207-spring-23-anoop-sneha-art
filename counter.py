from cmath import sin
import matplotlib as plt
import numpy as np
import cv2
import os
import math
from skimage import data, filters
from os import listdir
from os.path import isfile, join

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

#Thesea re the parameters that we are going to use
accum_res  = 5 # image resolution/accum resolution
min_between = 15 #Min dist between circles. 
minRadius = 610 #Min radius of a circle. 
maxRadius= 750 #The bigest circle expected
Canny_thr = 800 #anything above that is an edge automatically in Canny, the lower threshold is half of that.
Accum_thr = 900 #accumulator threshold for the circle centers at the detection stage

params_Hough = [accum_res, min_between, Canny_thr, Accum_thr, minRadius, maxRadius]

def ScaleImage(image):
    #resize image to ~1500x1500
    Width = image.shape[1]
    Scale = Width/1500
    new_size = (int(image.shape[1]/Scale), int(image.shape[0]/Scale)) 
    img_resized = cv2.resize(image, new_size )
    return img_resized

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


def FindCircles(params, scaled_img):

    #Convert to grayscale
    img_gr = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    
    #Enhance contrast
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+120))    ))

    img_contrast = cv2.LUT(img_gr, lookUpTable)    
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
    Finds 100 points on circle cnetered at "center" and with radius "radius".
    Sums up intensities of the "image" at these points. "band_width" is not used at the moment
    Essentially an integral over circumference. 
    '''

    #Calcualte values of sin and cos for each angle we will use, keep in in np.array fro speed
    sin_cos = np.array(  [   (math.sin(angle), math.cos(angle))    for angle in np.arange(0, 2*math.pi, math.pi/50)     ]        )
    
    #Find the x and y coordinates of points on the circle, summ up intensities for all these points
    sum_intensities = 0
    for value in sin_cos:

        x_at_angel = center[0]+ int(radius*value[1])
        if x_at_angel>=image.shape[1]: x_at_angel = image.shape[1]-1

        y_at_angel = center[1]+ int(radius*value[0])        
        if y_at_angel>=image.shape[0]: y_at_angel = image.shape[0]-1
        
        sum_intensities+=image[y_at_angel,x_at_angel]

    return sum_intensities

def Deriv_Intensity_f_R(image= np.array, center = (int, int),  min_radius = int, max_radius = int, step = int):
    
    #First Calucalte how integral brightness of the circles dpends on their radius
    #Do that starting from fairly large radius, to start close to the edge already
    Intensity_f_R = [   Circ_Integral(image, center, i, step)    for i in range(min_radius, max_radius, step) ]
    
    # Now calculate derivative of this function
    
    Deriv = []
    for i in range(0, len(Intensity_f_R)-1):
        Deriv.append(Intensity_f_R[i+1]-Intensity_f_R[i])

    #return this reriative
    return Deriv

def FindBestCircle(circles, image):
    #Convert to grayscale
    img_gr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Enhance contrast, but not as dramatically as for Hugh transform
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = int(   255/  ( 1  + math.exp (0.15*(-i+120))    ))

    img_contrast = cv2.LUT(img_gr, lookUpTable)    

    #Lets expand array of circles to store additional infor we find out
    #[0] - x, [1] - y, [2] - radius, [3] - MaxRadius, [4] - MaxDeriv
    zeros = np.zeros((circles.shape[0], 3), dtype = int)
    circles = np.concatenate(  (circles, zeros),  axis = 1)  

    for circle in circles:
        #Find the max radius that can possibly be at this cneter point
        #For this find how far away this point is from the center of the image
        #Keep in mind that image.shape[height, width], but circle[x,y,r]
        Y_offset = abs(image.shape[0]/2-circle[1])
        X_offset = abs(image.shape[1]/2-circle[0])
        max_X = int(image.shape[1]/2-X_offset)
        max_Y = int(image.shape[0]/2-Y_offset)
        max_R = min(max_X, max_Y)
        circle[3]=max_R
        
        #Calculate how intensity changes (i.e. derivative) as the radius increases.       
        Deriv_Step = 10
        Deriv_f_R = Deriv_Intensity_f_R(img_contrast,  (circle[0], circle[1]), 500, max_R, Deriv_Step)
        #Find what was the sharpest change for this circle, 
        MaxDeriv = max(Deriv_f_R)
        circle[4] =  MaxDeriv
        #Find where this sharp change occured
        R_of_Max = 500 + Deriv_f_R.index(MaxDeriv)*Deriv_Step
        circle[5] = R_of_Max

    #Find the circle that had the most abrupt change.
    #The idea is that as we go out from the most central point will have all LEDs come into view at once
    #As opposed to point that is off-center, where expanding circle wil hit only few LEDs at a time.
    #For that, sort all circles by the maxDrevi Value
    sorted_circles = circles[np.argsort(circles[:, 4])]
    BestCircle = sorted_circles[-1]

    #Now that we know where is the true center of the ROI, let's find its true radius.
    #Get full Deriv for this one circle, all the way to the max possible radius
    Deriv_Brightest = Deriv_Intensity_f_R(img_contrast,  (BestCircle[0], BestCircle[1]), 500, BestCircle[3], 5)

    #Add center coordinate to the list that we wil return
    return_value = [BestCircle[0], BestCircle[1]]

    #Go through the deriv and if the point and the next point is above 1500, say that is where ROI stops
    #Don't forget that we only calculate starting from 500 pixels from the center, so add those 500px
    for i in range(len(Deriv_Brightest)-1):
        if Deriv_Brightest[i]>1300   and Deriv_Brightest[i+1]>1300:
            return_value.append(500+i*5)
            break
    #If this condition was not met, just go and find were derivative was higher than 1500 at least once
    else:
        for i in range(len(Deriv_Brightest)-1):
            if (Deriv_Brightest[i-1]+Deriv_Brightest[i]+Deriv_Brightest[i+1]) > 1500:
                return_value.append(500+(i-1)*5)
                break
    
    return [return_value]

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
    '''
    I modified this code from https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
    '''
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
#imReference = cv2.imread(folder_path+'/'+'Ref.jpg', cv2.IMREAD_COLOR)

def main():
    #get a list of files in the folder with pics
    folder_path = os.path.dirname(__file__)+'/Smpl_Im'
    processed_path = os.path.dirname(__file__)+'/Thresholded'
    file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    for file in file_names:
        #Read the image
        img = cv2.imread(folder_path+'/'+file)

        #Scale Image
        img_scaled = ScaleImage(img)

        #Take scaled image, find circles and return an array of circles
        Circles, leveled = FindCircles(params_Hough, img_scaled)
        print(f"{file[0:-4]} {len(Circles):10}")

        #If there are any circles, filter them by centricity and dI/dR
        if len(Circles)>0:
            # In the list of centers, find the center of the true ROI
            # ROI is the circle with the sharpest brigtnest change
            # Brightness = integral of intensity over circumference
            # Brightness change = its derivative on radius
            # I.e. df/dr, where f = integral(intesity)*d(circ) 
            BestCirc = FindBestCircle(Circles, img_scaled)

            #Draw the best circel on the image
            #DrawCircles(BestCirc, img_scaled)

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
                ROI_img = ScaleImage(cut_image)

                #Enhance contrast
                final = EnhanceContrast(ROI_img, -20, 45)

                #Write final file to disk
                cv2.imwrite(processed_path+'/'+file, final)
        
        #Write the image with circles
        #cv2.imwrite(processed_path+'/'+file, img_scaled)

if __name__ == '__main__':
    main()

