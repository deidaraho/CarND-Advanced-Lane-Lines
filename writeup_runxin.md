## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/writeup_calibration.png "Calibration"
[image2]: ./output_images/writeup_calibration_test2.png "Road Transformed"
[image3]: ./output_images/bird-eye.png "Bird Eye"
[image4]: ./output_images/channels.png "RGB-HSL-LAB Channels"
[image5]: ./output_images/lanes_tests.png "Lanes in Test Images"
[image6]: ./output_images/window_polyfit.png "Window Polyfit"
[image7]: ./output_images/window_prefit.png "Pre-Polyfit"
[image8]: ./output_images/draw_data.png "Draw with Radius"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./pipeline.ipynb", starting from the comment "step one: calibration". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  
`imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result below, where the left is original and the right is undistored by calibration.

[calibration][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one below, where the left is original camera view and the right is the undistorted.
[Road Transformed][image2]

The demo picture above shows the difference between original and undistorted is tiny, however, the lane finding performance is improved by calibration.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at IN Section 8 through 24 in `pipeline.ipynb`).  Here's an example of my output for this step.  

Firstly, I transform to the bird eye view in front of the car as following,
[Bird Eye][image3]

Then, I tried the lane selection and segmentation performance in RGB, HSV and LAB channels, following are the related channel transforms from the same bird-view image,
[RGB-HSV-LAB Channels][image4]

After tuning threshold parameters and testing, I decided to use HLS L-channel and LAB B-channel for lane segmentation.
Following are the selection performance in testing images.
[Lane Binary Mask on Testing Images][image5]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarp()`, which appears in the 3rd code cell of the IPython notebook, pipeline.ipynb.  The `unwarp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(575, 464),
                  (750, 464), 
                  (258, 670), 
                  (1100, 670)])
dst = np.float32([(450, 0),
                  (w-450, 0),
                  (450, h),
                  (w-450, h)])
```


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
[Bird Eye][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The functions "sliding_window_polyfit" and "polyfit_using_prev_fit", which identify lane lines and fit a second order polynomial to both right and left lane lines, are clearly labeled in pipeline.ipynb. 
The first of these computes a histogram of the bottom half of the image and finds the bottom-most x position (or "base") of the left and right lane lines. 
Originally these locations were identified from the local maxima of the left and right halves of the histogram, but in my final implementation I changed these to quarters of the histogram just left and right of the midpoint. This helped to reject lines from adjacent lanes. 
The function then identifies ten windows from which to identify lane pixels, each one centered on the midpoint of the pixels from the window below. This effectively "follows" the lane lines up to the top of the binary image, and speeds processing by only searching for activated pixels over a small portion of the image. Pixels belonging to each lane line are identified and the Numpy polyfit() method fits a second order polynomial to each set of pixels. The image below demonstrates how this process works:
[Window Polyfit][image6]

The "polyfit_using_prev_fit" function performs basically the same task, but alleviates much difficulty of the search process by leveraging a previous fit (from a previous video frame, for example) and only searching for lane pixels within a certain range of that fit. The image below demonstrates this - the green shaded area is the range from the previous fit, and the yellow lines and red and blue pixels are from the current image:
[Pre-Polyfit][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in Cell 53 through 54 in my code in 'pipeline.ipynb'. The euqations are directly following tutorial.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in Cell 55 through 58 in my code in `pipeline.ipynb` in the function `draw_lane()` and 'draw_data()'.  Here is an example of my result with estimated radius on a test image:

[Lane with Radius][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The problems I encountered were almost exclusively due to lighting conditions, shadows, discoloration, etc. 
It took me some time to pick up channels and tune threshold parameters. Finally I discovered that the B channel of the LAB is able to isolate yellow lines very well. Still, I am wondering the robust of my pipeline in challenge vedios.

Several possible approaches may make my pipeline more robust, such as, more dynamic thresholding or even a learning-based thresholding and channel selection method.
