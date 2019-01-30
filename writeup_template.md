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
[image13]: ./output_images/13_binnary_warped_oppened.jpg "Openning"
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

The code for this step is contained in the 3rd and 5th cells of the jupyter notebook located in "/P2_advanced_line_detection.ipynb". The process to calibrate the image uses a chessboard pattern and starts with the definition of object points (known points on the cheese pattern in the real world given in x-y-z coordinates, where z=0 since the pattern is on a flat surface) and the image points (points found on the image, using the function: cv2.findChessboardCorners). These points are stored in numpy arrays for everyone of the images that are in the folder "/camera_cal". By matching image and object points (using the function cv2. calibrateCamera), we can get the camera matrix and the distortion parameters, which will allow us to undistort images using the "cv2.undistort" function. These parameters are then saved on a pickle file for future use.

![alt text][image1]
fig1. Distorted Chessboard pattern

![alt text][image2]
fig2. Undistorted Chessboard pattern

### Pipeline (single images)
Applying camera undistort, we can go from a distorted image:
![alt text][image5]
to an undistorted image:
![alt text][image5]
#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
