import numpy as np
import collections
import cv2
import pickle

from moviepy.editor import VideoFileClip

#################### class definitions #####################################

class LaneLine():
    """Define a class to receive the characteristics of each line detection"""
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of fit
        self.fitx = []
        # x values averaged
        self.avgx = []
        #polynomial coefficients in world coordinates
        self.fit_cr = [np.array([False])]
        #running weighted average of polynomial coefficients
        self.avg = [0, 0, 0]
        #polynomial coefficients in image coordinates
        self.fit = [np.array([False])]
        # last 5 polynomial coeffients
        self.c0 = collections.deque(maxlen=5)
        self.c1 = collections.deque(maxlen=5)
        self.c2 = collections.deque(maxlen=5)
        # radius of curvature at top, bottom
        self.r_top = 0
        self.r_bottom = 0
        # x position at top,bottom
        self.x_top = 0
        self.x_bottom = 0
        self.high_confidence = False

###################### reconstitute previously derived data ##################

def lens_factors():
    """get previously computed lens distortion factors"""
    dist_pickle = pickle.load(open("camera_cal_pickle.p", "rb"))
    return dist_pickle["mtx"], dist_pickle["dist"]

def warp_matrices():
    """compute transformations from camera view to bird's eye view and back"""

    # lane line center points from source image
    left_top = [596, 450]
    right_top = [689, 450]
    left_bottom = [300, 660]
    right_bottom = [1020, 660]
    src_pts = [left_bottom, left_top, right_top, right_bottom]
    src = np.float32(src_pts)

    # axis-oriented rectangle for target image to produce "bird's eye view"
    left, right, top, bottom = 500, 700, 20, 700
    tgt_pts = [[left, bottom], [left, top], [right, top], [right, bottom]]
    tgt = np.float32(tgt_pts)

    # compute transform for warp (and back)
    m = cv2.getPerspectiveTransform(src, tgt)
    minv = cv2.getPerspectiveTransform(tgt, src)
    return m, minv

#################### helper functions ##############################

def undistort(image):
    """correct image for camera lens distortion"""
    dst = cv2.undistort(image, MTX, DIST, None, MTX)
    return dst

def warp_image(image):
    """warp image to bird's eye view"""
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def threshold(image, s_thresh=(170, 255), sx_thresh=(20, 100), dir_thresh=(0.7, 1.3)):
    """apply image space conversion and Sobel gradient thresholding"""
    img = np.copy(image)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1) # Take the derivative in y
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal

    # Threshold gradient direction
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    gradbinary = np.zeros_like(absgraddir)
    gradbinary[(absgraddir >= dir_thresh[0]) & (absgraddir <= dir_thresh[1])] = 1

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold s color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Form binary combination
    combined_binary = np.zeros_like(sxbinary)
    #combined_binary[(gradbinary == 1) & ((s_binary == 1) | (sxbinary == 1))] = 1
    combined_binary[((s_binary == 1) | (sxbinary == 1))] = 1
    return combined_binary

def find_lines_using_prior(nonzerox, nonzeroy, left, right):
    """ use previously computed polynomial fit as starting point in search"""

    margin = 25 # width of the windows +/- margin

    left_lane_inds = ((nonzerox > (left.avg[0]*(nonzeroy**2) + left.avg[1]*nonzeroy + \
        left.avg[2] - margin)) & (nonzerox < (left.avg[0]*(nonzeroy**2) + \
        left.avg[1]*nonzeroy + left.avg[2] + margin)))

    right_lane_inds = ((nonzerox > (right.avg[0]*(nonzeroy**2) + right.avg[1]*nonzeroy + \
         right.avg[2] - margin)) & (nonzerox < (right.avg[0]*(nonzeroy**2) + \
         right.avg[1]*nonzeroy + right.avg[2] + margin)))

    return left_lane_inds, right_lane_inds

def find_lines_initially(binary_warped, nonzerox, nonzeroy, left, right):
    """search for lane line pixels in image"""

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9 # number of sliding windows
    window_height = np.int(binary_warped.shape[0]/nwindows) #  height of window

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100 # width of the windows +/- margin
    minpix = 50 # minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return left_lane_inds, right_lane_inds

def weighted_avg(left, right):
    """Compute weighted average of last 5 polynomial coefficients"""

    append_count = 5
    if left.detected:
        append_count = 1

    for i in range(append_count):
        left.c0.append(left.fit[0])
        left.c1.append(left.fit[1])
        left.c2.append(left.fit[2])
        right.c0.append(right.fit[0])
        right.c1.append(right.fit[1])
        right.c2.append(right.fit[2])
    
    #COEF = [1.0, 1.0, 2.0, 3.0, 5.0] # fibonacci sequence
    COEF = [1.0, 1.0, 1.0, 1.0, 1.0] # fibonacci sequence
    DENOM = np.sum(COEF)

    left.avg[0] = np.dot(left.c0, COEF)/DENOM
    left.avg[1] = np.dot(left.c1, COEF)/DENOM
    left.avg[2] = np.dot(left.c2, COEF)/DENOM
    right.avg[0] = np.dot(right.c0, COEF)/DENOM
    right.avg[1] = np.dot(right.c1, COEF)/DENOM
    right.avg[2] = np.dot(right.c2, COEF)/DENOM

def high_confidence(left, right):
    top_width = right.fitx[0] - left.fitx[0]
    bottom_width = right.fitx[-1] - left.fitx[-1]
    ratio = top_width/bottom_width
    left.high_confidence = (ratio > 0.9 and ratio < 1.1)
    right.high_confidence = left.high_confidence
    return right.high_confidence

def find_lane_lines(binary_warped, left, right):
    """find lane line pixels in binary_warped and return polynomial fit"""

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    if left.detected:
        left_lane_inds, right_lane_inds = find_lines_using_prior(nonzerox, nonzeroy, left, right)
    else:
        left_lane_inds, right_lane_inds = find_lines_initially(binary_warped, nonzerox, nonzeroy, left, right)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left.fit = np.polyfit(lefty, leftx, 2)
    right.fit = np.polyfit(righty, rightx, 2)

    left.fitx = left.fit[0]*ploty**2 + left.fit[1]*ploty + left.fit[2]
    right.fitx = right.fit[0]*ploty**2 + right.fit[1]*ploty + right.fit[2]

    # Conditionally update weighted average of polynomial coefficents 
    if high_confidence(left, right):
        weighted_avg(left, right)

    # Fit new polynomials to x,y in world space
    left.avgx = left.avg[0]*ploty**2 + left.avg[1]*ploty + left.avg[2]
    right.avgx = right.avg[0]*ploty**2 + right.avg[1]*ploty + right.avg[2]

    left.fit_cr = np.polyfit(ploty*ym_per_pix, left.avgx*xm_per_pix, 2)
    right.fit_cr = np.polyfit(ploty*ym_per_pix, right.avgx*xm_per_pix, 2)

    # use fit to optimize search in next frame
    left.detected = True
    right.detected = True

def unwarp_image_with_lane(image, left, right):
    """draw lane in bird's eye view then unwarp and merge with original view"""

    # Create an image to draw the lines on
    warp = np.zeros_like(image).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    avg_pts_left = np.array([np.transpose(np.vstack([left.avgx, ploty]))])
    avg_pts_right = np.array([np.flipud(np.transpose(np.vstack([right.avgx, ploty])))])
    avg_pts = np.hstack((avg_pts_left, avg_pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp, np.int_([avg_pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (MINV)
    unwarp = cv2.warpPerspective(warp, MINV, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    merge = cv2.addWeighted(image, 1, unwarp, 0.3, 0)
    #merge = cv2.addWeighted(image, 1, warp, 0.3, 0)

    #pts_left = np.array([np.transpose(np.vstack([left.fitx, ploty]))])
    #pts_right = np.array([np.flipud(np.transpose(np.vstack([right.fitx, ploty])))])

    #left_lane_pts = np.array(pts_left.reshape(-1, 1, 2), np.int32)
    #right_lane_pts = np.array(pts_right.reshape(-1, 1, 2), np.int32)

    #cv2.polylines(merge, [left_lane_pts], False, (255, 0, 0), thickness=3)
    #cv2.polylines(merge, [right_lane_pts], False, (255, 0, 0), thickness=3)

    return merge

def radius_and_offset(left, right, y_eval):
    """compute radius of curvature and lane centerline offset at y=0 and y=y_eval"""

    # Calculate the radii of curvature at top
    y = 0
    left.r_top = ((1 + (2*left.fit_cr[0]*y + left.fit_cr[1])**2)**1.5) / np.absolute(2*left.fit_cr[0])
    right.r_top = ((1 + (2*right.fit_cr[0]*y + right.fit_cr[1])**2)**1.5) / np.absolute(2*right.fit_cr[0])

    # compute lane line x position
    left.x_top = left.fit_cr[0]*y**2 + left.fit_cr[1]*y + left.fit_cr[2]
    right.x_top = right.fit_cr[0]*y**2 + right.fit_cr[1]*y + right.fit_cr[2]

    # Calculate the radii of curvature at bottom
    y = y_eval*ym_per_pix
    left.r_bottom = ((1 + (2*left.fit_cr[0]*y + left.fit_cr[1])**2)**1.5) / np.absolute(2*left.fit_cr[0])
    right.r_bottom = ((1 + (2*right.fit_cr[0]*y + right.fit_cr[1])**2)**1.5) / np.absolute(2*right.fit_cr[0])

    # compute lane line x position
    left.x_bottom = left.fit_cr[0]*y**2 + left.fit_cr[1]*y + left.fit_cr[2]
    right.x_bottom = right.fit_cr[0]*y**2 + right.fit_cr[1]*y + right.fit_cr[2]

def annotate_image(image, left, right):
    """Add curvature and offset annotations"""

    # avg radius and lane center offset at bottom
    avg_radius = int(0.5*(left.r_bottom + right.r_bottom)) # average left and right
    ctr = 0.5*(left.x_bottom + right.x_bottom)
    offset = ctr - 640*xm_per_pix

    w_top = right.x_top - left.x_top
    w_bottom = right.x_bottom - left.x_bottom

    r_l_string = 'Left Radius of curvature = {} m'.format(int(left.r_bottom))
    r_r_string = 'Right Radius of curvature = {} m'.format(int(right.r_bottom))
    t_w_string = 'Lane width at top = {:.2f} m'.format(w_top)
    b_w_string = 'Lane width at bottom = {:.2f} m'.format(w_bottom)
    ratio_string = 'Lane width ratio = {:.2f} m'.format(w_top/w_bottom)
    confidence_string = 'High confidence = {}'.format(left.high_confidence)

    curvature_string = 'Radius of curvature = {} m'.format(avg_radius)
    offset_string = 'Vehicle is {:.2f} m '.format(np.abs(offset))

    if offset > 0:
        offset_string += 'right of center'
    else:
        offset_string += 'left of center'

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, curvature_string, (20, 50), font, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, offset_string, (20, 100), font, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, r_l_string, (20, 150), font, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, r_r_string, (20, 200), font, 1, (0,255,255), 2, cv2.LINE_AA)
    #cv2.putText(image, t_w_string, (20, 250), font, 1, (0,255,255), 2, cv2.LINE_AA)
    #cv2.putText(image, b_w_string, (20, 300), font, 1, (0,255,255), 2, cv2.LINE_AA)
    #cv2.putText(image, ratio_string, (20, 350), font, 1, (0,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, confidence_string, (20, 400), font, 1, (0,255,255), 2, cv2.LINE_AA)

    return image

####################### image processing pipeline #################################

def process_image(image):
    """image processing pipeline"""

    # camera lens distortion correction
    image = undistort(image)

    # perspective transform to bird's eye view
    warped = warp_image(image)

    # apply thresholding to create binary image for lane line identification
    binary_warped = threshold(warped)

    # find lane line pixels in binary_warped and fit polynomial
    global left, right
    find_lane_lines(binary_warped, left, right)

    # draw lane in warped perspective and unwarp back
    rgb_warped = np.dstack((binary_warped, binary_warped, binary_warped))*255 # DEBUG
    image = unwarp_image_with_lane(image, left, right)
    #image = unwarp_image_with_lane(rgb_warped, left, right) # DEBUG

    # compute radius of curvature and lane centerline offset
    radius_and_offset(left, right, image.shape[0])

    # annotate image
    image = annotate_image(image, left, right)

    return image
    #rgb_warped = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #return rgb_warped

########################################################################

# set up constants previously computed in notebook code
MTX, DIST = lens_factors() # camera lens distortion correction
M, MINV = warp_matrices()  # warping transform and inverse

height = 720
ploty = np.linspace(0, height-1, height)
 
# Define conversions in x and y from pixels space to meters
ym_per_pix = 3.0/85 # meters per pixel in y dimension
xm_per_pix = 3.7/275 # meters per pixel in x dimension

# global variables
left = LaneLine()
right = LaneLine()

# load video, process each frame, write processed video
input_video = VideoFileClip("project_video.mp4")#.subclip(20, 25)
output_video = input_video.fl_image(process_image)
output_video.write_videofile('test_output.mp4', audio=False)
