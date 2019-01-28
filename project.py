import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

#================ Load Camera Calibration================
pickle_in = open("camera_cal/wide_dist_pickle.p","rb")
camera_calibration = pickle.load(pickle_in)
mtx = camera_calibration['mtx']
dist = camera_calibration['dist']

#============= Load Warp and Unwarp matrixes ============
pickle_in = open("camera_cal/warp_pickle.p","rb")
warp_matrixes = pickle.load(pickle_in)
M = warp_matrixes['m']
M_inv = warp_matrixes['m_inv']
print(M)
print(M_inv)


#===============Functions to apply gradient thresholding===========
def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(50, 100)):
    
    # Apply the following steps to img
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if(orient == 'x'):
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    sobel = pow((pow(sobelX,2) + pow(sobelY,2)),1/2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*sobel/np.max(sobel))
    # 5) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

def dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3)):
    
    # 2) Take the gradient in x and y separately
    sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelX)
    abs_sobely = np.absolute(sobelY)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_gradient = np.arctan2(abs_sobely, abs_sobelx) 
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(dir_gradient)
    sbinary[(dir_gradient >= thresh[0]) & (dir_gradient <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

def color_threshold(img, s_thresh=(170, 255)):
    # Threshold color channel
    s_binary = np.zeros_like(img)
    s_binary[(img >= s_thresh[0]) & (img <= s_thresh[1])] = 1
    return s_binary

#================Function to get warped binary images============
def get_warped_binary_S(img):
    img_size = (img.shape[1], img.shape[0])
    #undistort image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    #change to hls color space
    hls_img = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HLS)
    S_img = hls_img[:,:,2]
    #x edges
    S_sobel_X = abs_sobel_thresh(S_img, orient='x', thresh=(10, 200))
    S_sobel_Y = abs_sobel_thresh(S_img, orient='y', thresh=(10, 200))
    #xy edges
    S_sobel_XY = mag_thresh(S_img, mag_thresh=(10, 200))
    #dir edges
    S_sobel_Dir = dir_threshold(S_img, thresh=(0.5, 1.3))
    #combined
    S_comb = np.zeros_like(S_img)
    S_comb[((S_sobel_X == 1) & (S_sobel_Y == 1)) | ((S_sobel_XY == 1) & (S_sobel_Dir == 1))] = 1
    #S_comb[((S_sobel_X == 1) & (S_sobel_Y == 1))] = 1
    #Color thresh
    S_col = color_threshold(S_img, s_thresh=(160, 255))
    #Combine colors and gradient
    S_combined_2 = np.zeros_like(S_img)
    S_combined_2[(S_comb == 1) | (S_col == 1)] = 1
    #Perspective transform
    warped_binary2 = cv2.warpPerspective(S_combined_2, M, img_size, flags=cv2.INTER_LINEAR)
    return warped_binary2

def get_warped_binary_V(img):
    img_size = (img.shape[1], img.shape[0])
    #undistort image
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    #change to hls color space
    hsv_img = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2HSV)
    V_img = hsv_img[:,:,2]
    V_img_blur = cv2.GaussianBlur(V_img, (5, 5), 0)
    #x edges
    V_sobel_X = abs_sobel_thresh(V_img, orient='x', thresh=(30, 100))
    V_sobel_Y = abs_sobel_thresh(V_img, orient='y', thresh=(60, 100))
    #xy edges
    V_sobel_XY = mag_thresh(V_img, mag_thresh=(50, 100))
    #dir edges
    V_sobel_Dir = dir_threshold(V_img, thresh=(0.5, 1.3))
    #combined
    V_comb = np.zeros_like(V_img)
    V_comb[((V_sobel_X == 1) & (V_sobel_Y == 1)) | ((V_sobel_XY == 1) & (V_sobel_Dir == 1))] = 1
    #V_comb[((V_sobel_X == 1) & (V_sobel_Y == 1))] = 1
    #Color thresh
    V_col = color_threshold(V_img, s_thresh=(230, 255))
    #Combine colors and gradient
    V_combined_2 = np.zeros_like(V_img)
    V_combined_2[(V_comb == 1) | (V_col == 1)] = 1
       
    #Perspective transform
    warped_binary2 = cv2.warpPerspective(V_combined_2, M, img_size, flags=cv2.INTER_LINEAR)
    #perform openning morfological operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    opening = cv2.morphologyEx(warped_binary2, cv2.MORPH_OPEN, kernel) 
    return warped_binary2

#=====================Find lines and calculate their curvature============
# def measure_curvature_real(left_fit,right_fit,ploty):
#     '''
#     Calculates the curvature o
#     f polynomial functions in meters.
#     '''
#     ym_per_pix = 30/600 # meters per pixel in y dimension
#     xm_per_pix = 3.7/700 # meters per pixel in x dimension
#     # Define y-value where we want radius of curvature
#     # We'll choose the maximum y-value, corresponding to the bottom of the image
#     y_eval = np.max(ploty)-100 * ym_per_pix
    
#     ##### Implement the calculation of R_curve (radius of curvature) #####
#     left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
#     right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
#     return left_curverad, right_curverad

#Functions to Fit a 2nd order polynom to the binary warped image
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    dead_zone_x = histogram.shape[0]//6
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[dead_zone_x:midpoint]) + dead_zone_x
    rightx_base = np.argmax(histogram[midpoint:histogram.shape[0]-dead_zone_x]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 200
    # Set minimum number of pixels found to recenter window
    minpix = 20

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = int(leftx_current - margin/2) # Update this
        win_xleft_high = int(leftx_current + margin/2)  # Update this
        win_xright_low = int(rightx_current - margin/2)  # Update this
        win_xright_high = int(rightx_current + margin/2)  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if(len(good_left_inds) > minpix):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, drawEnable = False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    cv2.polylines(out_img,  np.int32(pts_left),  False,  (0, 255, 0),  thickness=5)
    cv2.polylines(out_img,  np.int32(pts_right),  False,  (0, 255, 0),  thickness=5)
    if(drawEnable):
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty


def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit, drawEnable = False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 130

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    if(drawEnable):
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, left_fit, right_fit, ploty

def draw_area(img, left_fit, right_fit, ploty):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    warp_zero = np.zeros_like(gray).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    pts_left_line = np.vstack((left_fitx,ploty)).astype(np.int32).T
    pts_right_line = np.vstack((right_fitx,ploty)).astype(np.int32).T
    cv2.polylines(color_warp,  np.int32(pts_left),  False,  (255, 0, 0),  thickness=20)
    cv2.polylines(color_warp,  np.int32(pts_right),  False,  (0, 0, 255),  thickness=20)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.8, 0)
    return color_warp, result

#=====================================================================================
#=========== Define a class to receive the characteristics of each line detection ====
#=====================================================================================
class Lines():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='int')  
        self.last_best_fit = np.array([0,0,0], dtype='int')   
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = 0.0
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        self.ploty = np.array([0,0,0], dtype='int') 
        ##=============
        self.filter_window = None
        self.measures = np.empty((0,3), int)
        self.wrong_count = 0
    
    def measure_curvature_real(self, actual_fit, ploty):
        '''
        Calculates the curvature o
        f polynomial functions in meters.
        '''
        ym_per_pix = 30/700 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty) * ym_per_pix
        print(y_eval)
        
        ##### Implement the calculation of R_curve (radius of curvature) #####
        curvature = ((1 + (2*actual_fit[0]*y_eval + actual_fit[1])**2)**1.5) / np.absolute(2*actual_fit[0])
        return curvature

    def add_measure(self, measure, ploty):

        #Calculate radius of curvature
    
        if(self.radius_of_curvature == 0):
            actual_curvature = self.measure_curvature_real(measure, ploty)
            self.radius_of_curvature = actual_curvature
            self.measures = np.append(self.measures, np.array([measure]), axis=0)
        else:
            self.measures = np.append(self.measures, np.array([measure]), axis=0)
            #calculate curvature, x distance, parallelism, check parabpa formula: shift in x and y and curvature
            if self.measures.shape[0] > 6:
                self.measures = np.delete(self.measures, (0), axis=0)
        self.best_fit = np.average(self.measures, axis=0)
        if(self.best_fit[0] == 0):
            self.radius_of_curvature = self.measure_curvature_real(self.last_best_fit, ploty)
        else:
            self.radius_of_curvature = self.measure_curvature_real(self.best_fit, ploty)
        #self.best_fit = measure

lane_left = Lines()
lane_right = Lines()
#15s - 21-24 - 39-40

def horizontal_distance(left_fit,right_fit,ploty):
    xm_per_pix = 3.7/700
    left_fitx = (left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]) * xm_per_pix
    right_fitx = (right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]) * xm_per_pix
    average_distance = np.average(right_fitx - left_fitx)
    std_distance = np.std(right_fitx - left_fitx)
    
    x_der = right_fitx[0]
    x_izq = left_fitx[0]
    center_car = (1280*xm_per_pix/2.0)
    center_road = ((x_der+x_izq)/2.0)
    position = center_car-center_road
    #print("der: " + "{:.2f}".format(x_der) + " | izq: " + "{:.2f}".format(x_izq), " | center_car: " + "{:.2f}".format(center_car),  " | center_road: " + "{:.2f}".format(center_road) )

    return average_distance, std_distance, position

def measure_curvature_real(actual_fit, ploty):
    '''
    Calculates the curvature o
    f polynomial functions in meters.
    '''
    ym_per_pix = 30/700 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty) * ym_per_pix
    #print(y_eval)
        
    ##### Implement the calculation of R_curve (radius of curvature) #####
    curvature = ((1 + (2*actual_fit[0]*y_eval + actual_fit[1])**2)**1.5) / np.absolute(2*actual_fit[0])
    return curvature

def advanced_process_image(img):
    global lane_left
    global lane_right
    error = False

    if(lane_right.wrong_count >= 3):
        lane_right.lasst_best_fit = lane_right.best_fit
        lane_right.best_fit = np.array([0,0,0], dtype='int')
        lane_right.radius_of_curvature = 0.0 
        lane_right.measures = np.empty((0,3), int)

    if(lane_left.wrong_count >= 5):
        lane_left.lasst_best_fit = lane_right.best_fit
        lane_left.best_fit = np.array([0,0,0], dtype='int')
        lane_left.radius_of_curvature = 0.0 
        lane_left.measures = np.empty((0,3), int)

    binary_warped = get_warped_binary_S(img)

    if(lane_left.detected and lane_right.detected):
        out, left_fit, right_fit, ploty = search_around_poly(binary_warped, lane_left.best_fit, lane_right.best_fit)
    else:
        out, left_fit, right_fit, ploty = fit_polynomial(binary_warped)
        lane_left.detected = True
        lane_right.detected = True
    #measure horizontal distances and curvatures
    actual_left_curvature = measure_curvature_real(left_fit, ploty)
    actual_right_curvature = measure_curvature_real(right_fit, ploty)
    h_distance_avg, h_distancd_std, position = horizontal_distance(left_fit,right_fit,ploty)
    #print("=============================")
    #make decisions based on curvature and horizontal distance
    if( (h_distance_avg>3.0) and (h_distance_avg<4.2) and (h_distancd_std<0.3) ):
        if( (abs(actual_left_curvature / lane_left.radius_of_curvature) < 2.5) or lane_left.radius_of_curvature == 0.0 ):
            lane_left.add_measure(left_fit, ploty)
            lane_left.wrong_count = 0
        else:
            lane_left.wrong_count = lane_left.wrong_count + 1
            lane_left.detected = False
            #print("crazy left curvature")
            print(actual_left_curvature)
        
        if( (abs(actual_right_curvature / lane_right.radius_of_curvature) < 2.5) or lane_right.radius_of_curvature == 0.0 ):
            lane_right.add_measure(right_fit, ploty)
            lane_right.wrong_count = 0
        else:
            lane_right.wrong_count = lane_right.wrong_count + 1
            lane_right.detected = False
            #print("crazy right curvature")
            print(actual_right_curvature)
    else:
        lane_left.wrong_count = lane_left.wrong_count + 1
        lane_right.wrong_count = lane_right.wrong_count + 1
        #print("crazy horizontal distance")
        error = True
        lane_left.detected = False
        lane_right.detected = False
    
        
    
    #print(lane_left.best_fit)
    #print(ploty)
    if(lane_left.best_fit[0]== 0):
        if(lane_right.best_fit[0]== 0):
            color_warped, result = draw_area(img, lane_left.last_best_fit, lane_right.last_best_fit, ploty)
        else:
            color_warped, result = draw_area(img, lane_left.last_best_fit, lane_right.best_fit, ploty)
    else:
        if(lane_right.best_fit[0]== 0):
            color_warped, result = draw_area(img, lane_left.best_fit, lane_right.last_best_fit, ploty)
        else:
            color_warped, result = draw_area(img, lane_left.best_fit, lane_right.best_fit, ploty)

    #return cv2.cvtColor(binary_warped*255, cv2.COLOR_GRAY2BGR)
    message = "left_curv="+str(lane_left.radius_of_curvature)+"\n right curv="+str(lane_right.radius_of_curvature)+"\n h_dist="+str(h_distance_avg)+"\n std_dist="+str(h_distancd_std)
    return color_warped, out, binary_warped, result, lane_left.radius_of_curvature, lane_right.radius_of_curvature, h_distance_avg, h_distancd_std, lane_left.wrong_count, lane_right.wrong_count, error, position

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter("out_challenge.avi", fourcc, 25.0, (1280,720))
cap = cv2.VideoCapture('project_video.mp4')
curvature = 0.0
car_position = 0.0
alpha = 0.9
while(cap.isOpened()):
    ret, frame = cap.read()
    #get result and partial results
    color_warped, windows, binary_warped, result, left_curvature, right_curvature, h_distance_avg, h_distancd_std, left_wrong, right_wrong, error, position = advanced_process_image(frame)
    if(error):
        cv2.imwrite("error.jpg",frame)

    if(curvature != 0.0):
        curvature = (1-alpha)*((left_curvature+right_curvature)/2.0) + alpha*curvature
        car_position = (1-alpha)*position + alpha*car_position
    else:
        curvature = ((left_curvature+right_curvature)/2.0)
        car_position = position
    #add messages
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,"Radius of curvature: "+"{:.2f}".format(curvature)+"(m)",(20,70), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    if(car_position < 0):
        cv2.putText(result,"Car is: "+"{:.2f}".format(abs(car_position))+"m right of center",(20,210), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    else:
        cv2.putText(result,"Car is: "+"{:.2f}".format(abs(car_position))+"m left of center",(20,210), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"left_curv:"+"{:.2f}".format(left_curvature),(20,70), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"right_curv:"+"{:.2f}".format(right_curvature),(20,140), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"car_position:"+"{:.2f}".format(position),(20,210), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"h_dist:"+"{:.2f}".format(h_distance_avg),(20,280), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"std_dist:"+"{:.2f}".format(h_distancd_std),(20,350), font, 2.0,(255,255,255),2,cv2.LINE_AA)

    # cv2.putText(result,"left_wrong:"+str(left_wrong),(20,420), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(result,"right_wrong:"+str(right_wrong),(20,490), font, 2.0,(255,255,255),2,cv2.LINE_AA)
    out.write(result)
    #resize all images
    windows = cv2.resize(windows, (0, 0), None, .47, .47)
    color_warped = cv2.resize(color_warped, (0, 0), None, .47, .47)
    binary_warped = cv2.resize(255*binary_warped, (0, 0), None, .47, .47)
    result = cv2.resize(result, (0, 0), None, .47, .47)

    # Make the grey scale image have three channels
    binary_warped = cv2.cvtColor(binary_warped, cv2.COLOR_GRAY2BGR)

    # Join images
    line1 = np.hstack((windows, binary_warped))
    line2 = np.hstack((color_warped, result))
    images = np.vstack((line1,line2))
    

    

    #pause reproduction
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'):
        break
    else:
        if key == ord('p'):
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', images)
                if key2 == ord('p'):
                    break
        cv2.imshow('frame',images)


cap.release()
cv2.destroyAllWindows()