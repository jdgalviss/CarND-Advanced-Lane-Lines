# **Advanced Lane Finding** 

## Writeup - Juan David Galvis

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Confirm that detected lines are consistent with reality and previous lines, i.e. they have similar curvature, horizontal distance is around 3.7m, etc.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/01_original.jpg "Chessboard_Pattern"
[image2]: ./output_images/02_undistorted.jpg "Undistorted_Patternn"
[image3]: ./output_images/03_lines.jpg "Lines_Detected"
[image4]: ./output_images/04_warped.jpg "Bird-View"
[image5]: ./output_images/05_test_img.jpg "Test_Image"
[image6]: ./output_images/06_undistorted.jpg "Undistorted_Image"
[image7]: ./output_images/07_color_channels.png "Color_Channels"
[image8]: ./output_images/08_edges.png "Edges_Channels"
[image9]: ./output_images/09_color.png "Color_Threshold_Channels"
[image10]: ./output_images/10_mixed.png "Binary_Channels"
[image11]: ./output_images/11_warped.png "Binary_Warped_Channels"
[image12]: ./output_images/12_binary_warped.jpg "Binary_Warped_R_V"
[image13]: ./output_images/13_binary_warped_oppened.jpg "Openning"
[image14]: ./output_images/14_windows.jpg "Windows"
[image15]: ./output_images/15_color_warped.jpg "Color_Warped"
[image16]: ./output_images/16_result_windows.jpg "Windows_Result"
[image17]: ./output_images/17_around.jpg "Search_around"
[image18]: ./output_images/18_color_warped.jpg "Around_Warped"
[image19]: ./output_images/19_result_around.jpg "Around_Result"
[video1]: ./project_result.avi "Output Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Camera Calibration

The code for this step is contained in the 3rd and 5th cells of the jupyter notebook located in `/P2_advanced_line_detection.ipynb`. The process to calibrate the image uses a chessboard pattern and starts with the definition of object points (known points on the cheese pattern in the real world given in x-y-z coordinates, where z=0 since the pattern is on a flat surface) and the image points (points found on the image, using the function: cv2.findChessboardCorners). These points are stored in numpy arrays for everyone of the images that are in the folder "/camera_cal". By matching image and object points (using the function cv2. calibrateCamera), we can get the camera matrix and the distortion parameters, which will allow us to undistort images using the "cv2.undistort" function. These parameters are then saved on a pickle file for future use.

![alt text][image1]
fig1. Distorted Chessboard pattern

![alt text][image2]
fig2. Undistorted Chessboard pattern

### Pipeline
Here I present a summary of my pipeline for lane dataction.
#### 1. Distortion correction
Applying camera undistort (function undistort(img, mtx, dist, None, mtx)), we can go from a distorted image (img):
![alt text][image5]
to an undistorted image:
![alt text][image5]
by using the camera matrix (mtx) and the distortion coefficients(dist).


#### 2. Getting a thresholded binary image

I used a combination of color and gradient thresholds to generate a binary image (The different functions used for gradient and color thresholding are in the file `project.py` between lines 25 and 76).

In order to identify which channels were more appropiate to do the thresholding, I displayed all channels for several images:
![alt text][image7]

It was pretty clear that channels R from RGB, S from HLS and V from HSV are the ones where lane lines seem to be more highlited. So now, to define which ones I was actually going to use, I performed edge detection on these channels by combining X edges with Y edges and XY edges with edge-gradient thresholding:
![alt text][image8]

and color thresholding:
![alt text][image9]

By mixing, edge and color thresholding, from these channels we can get:
![alt text][image10]

which are clearly showing the lane lines as expected. The code used to get binary thresholded images on these 3 channels can be found in the file `/project.py` between lines 82 and 171

#### 3. Perspective transform
To get the perspective transform matrix and inverse matrix, I reused some of the code from the first project in order to get straight lane lines on an image that I knew contained straight lines:
![alt text][image3]

The source and destination points I got are the following:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 587, 455      | 300, 100      | 
| 230, 712      | 300, 720      |
| 1098, 712     | 980, 720      |
| 698, 455      | 980, 100      |

With these points I calculated the matrixes M and M_inv using the function `cv2.getPerspectiveTransform(src, dst)`

By taking the 4 points (2 points for each line) and defining the margins, I could transform (using the code found on the fifth block of the jupyter notebook `/P2_advanced_line_detection.ipynb`) the image from its normal perspective to a bird-eye view (using function cv2.warpPerspective()):
![alt text][image4]

After getting tha M and M_inv matrixes, I saved them in a pickle file for future use with the function .

Binary edges images can be then transformed to get a warped binary image like following:
![alt text][image11]

After trying with several images, I noticed that in some cases (shadows, change of color in the pavement), it was hard to find a proper threshold for the S channel that didn't produce extreme noise, this is why I decided to work with a combination of R and V channels. This combination produces binary warped images like following:

![alt text][image12]

To use dome of the noise produced by shadows I decided to use the morphological operation called openning:
![alt text][image13]

#### 4. Lane-line identification

The functions used to fit a 2nd order polynomial are in file `/project.py` betwwen lines 178 to 408. For the first image and in cases where the track is lost (determined by a sanity check) I used the a "windows" method:
![alt text][image14]

When the track is locked, it is better to simply detect lane lines around the previous polynomial:
![alt text][image17]

#### 5. Calculating curvature

I did this in lines 502 through 516 in my code in `/project.py` using the formula given in the lessons.
```python

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
 
    ##### Implement the calculation of R_curve (radius of curvature) #####
    curvature = ((1 + (2*actual_fit[0]*y_eval + actual_fit[1])**2)**1.5) / np.absolute(2*actual_fit[0])
    return curvature * actual_fit[0]/abs(actual_fit[0])
```

#### 6. Sanity check

In order to decide wether a detection should be taken into account I decided to check 3 basic things

1. Actual curvature should be similar to the curvature on the previous frame
2. Left and right line curves should at least have the same direction sign (second derivative), unless the lines are almost straight (high curvatures)
3. Lines should be relatively paralel. Here I measured horizontal distance average (it should be around 3.7m) and standard deviation in the following function:

```python
#====Function to measure horizontal position's average and standard deviation====#
#==========between lines. The function also measures the car's position==========#
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
    return average_distance, std_distance, position
```

#### 6. Result
Once lane lines are detected and pass the sanity check, curvature is calculated and lines are shown in the original image using an inverse perspective transform.


![alt text][image15]

![alt text][image16]

---

### Pipeline (video)
A video of the result
![alt text][video1]

Here's a [link to my video result](./project_result.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

#### Issues
The pipeline works pretty well with the project video, however when trying to use it with the challenge videos, it can be seen right away that the parameters that worked to get the binary image don't work that well now, producing false detections that would make the lanes detected useless. It seems like some sort of adaptive thresholds for the gradient and color thresholds might be useful. The same way, filtering the lines based on their location when there are multiple detection of lane lines could be of help.

In summary, the biggest problem is the robustness of the binary image generation, environments with different lighting, pavement color, shades and of course during the night would produce binary images from which it will be too hard and even impossible to extract the right lane lines.

#### Important features.
Perhaps the hardest part for the development of the pipeline was the election of correct thresholding parameters. 

Another key factor is the sanity check, without it, certain lanes that are false detected on few frames would strongly affect the performance of the detections.

Finally, by averaging the lane lines along a window of around 10 frames, the performance of the pipeline increased.
