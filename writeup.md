## Advanced Lane Finding Project

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

[image1]: ./output_images/image1.png "Chessboard distortion correction"
[image2]: ./output_images/image2.png "Highway distortion correction"
[image3.1]: ./output_images/image3.1.png "Threshold-S"
[image3.2]: ./output_images/image3.2.png "Stacked threshold and binary combination"
[image4]: ./output_images/image4.png "Warp example"
[image5]: ./output_images/image5.png "Fit visual"
[image6]: ./output_images/image6.png "Annotated output"
[video1]: ./project_video.mp4 "Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

#### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.   

You're reading it! Additional comments can be found in the IPython notebook, `AdvancedLaneFinding.ipynb`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first two code cells of the IPython notebook. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. Note that some of the calibration images clipped the chessboard pattern were discarded for the computation. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

For the video pipeline, I did not recompute the camera distortion parameters. Rather, I saved these in the file `camera_cal_pickle.p` and read them in the function `lens_factors()`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the third code cell of the IPython notebook. I reuse the previously computed and saved camera calibration data and apply the `cv2.undistort()` function to a test image. The following pair of images illustrate the camera lens distortion correction for an example test image:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in code cells 4 and 5).  The first pair of images show an example of color thresholding in the S channel (after conversion from RGB to HLS). This does a nice job of highlighting the lane lines, but the other cars and trees also are evident. Masking would obsure the trees, but masking would be less effective for cars in our lane or neighboring lanes. Moreover, the sliding window algorithm for lane line identification (discussed later) effectively masks features in the non-lane area. 

![alt text][image3.1]

I experimented with Sobel gradients in order to suppress the non-lane line information. The gradient direction thresholding output was very noisy (the red channel in the image on the left). I wondered if filtering for image gradients in the diagonal directions would be effective (using a Kirsch operator) but the solution I settled on was to AND the gradient direction threshold output with the OR of the S channel threshold output and x direction gradient threshold output (the image on the right).

![alt text][image3.2]

While this combination of thresholding produced good results for my test image, performance on some video frames was poor. I removed the gradient direction thresholding to achieve more consistent performance for the video. The code for the video pipeline is in the function `threshold()`.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The matrices for the perspective transform (and inverse) are intiialized in the function  `warp_matrices()`.  This function requires source (`src`) and target (`tgt`) points.  I chose the source and target points by selecting points defining a rectangle in the lane from the driving perspective and from above:

| Source        | Target        | 
|:-------------:|:-------------:| 
| 300, 660      | 500, 700      | 
| 596, 450      | 500, 20       |
| 689, 450      | 700, 20       |
| 1020, 660     | 700, 700      |

These matrices are used in the functions `warp_image()` to transform the image to the bird's eye view and `unwarp_image_with_lane()` to return to the original perspective.

I verified that my perspective transform was working as expected by drawing the `src` and `tgt` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. 

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Beginning with a histogram of the lower half of the warped binary image to identify the peaks corresponding to the lane lines, the function `find_lane_lines_intially()` walks up the image collecting pixels within a window of expected lane line position. The function `np.polyfit()` is used to fit a quadratic polynomial to the pixels (in `find_lane_lines()`). On subsequent frames the prior polynomial is used to accelerate the pixel collection (in `find_lane_lines_using_prior()`).

The pixel-space polynomials are evaluated at the top and bottom of the image so as to compare lane width. If the width is consistent, then the fit is considered good and this newest fit is averaged with the previous four successful fits to produce a composite fit. Otherwise, the newest fit is ignored. The composite fit is used to generate another set of lane line points. These pixel coordinates are mapped to world space coordinates and refit with another quadratic polynomial.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is computed from the quadratic polynomial used to fit the lane line. The curvature of each lane line is evaluated at the bottom of the image (nearest the car) and averaged. The offset is computed by averaging the x-coordinates when evaluating the polynomials at the bottom of the image and subtracting it from the vehicle center. This computation is done in the function `radius_and_offset()`.

The quadratics are converted from a pixel-space fit to a world-space fit by scaling by an estimated meters:pixel factor in x and y. In my case, the estimates seem suspect.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this in the function `unwarp_image_with_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach taken is very brittle - small changes in one frame will give bad results. Only by averaging successive frames and discarding outliers could I generate meaningful lane data.
- Finding the two highest peaks in the histogram is an unrealiable approach for the initial locating of the lane lines. Since we know the car (and therefore camera) is roughly centered in the lane, looking for the two relative peaks a lane width apart seems more robust.
- The quadratic fit to the pixels seems inadequate. Two important constraints are ignored: 1) the lane lines should be parallel in plan view and 2) the slope of the lines should vertical at the base of the image (parallel to the car's direction - assuming the car isn't spinning out!).
- My current implementation judges the correctness of the polynomial fit by comparing the lane width at the top and bottom of the frame. This is not very robust. The slope at the top and bottom should also be a factor. Also the number of pixels (and their spatial distribution) used to fit the quadratic should be considered.
- My current implementation also considers the fit correctness for the quadratics as a pair. This ignores the fact that typically only one quadratic fit is bad in any particular frame (for the test video at least).
- More tuning of the thresholding logic seems indicated. Perhaps using a Kirsch operator would improve performance.
