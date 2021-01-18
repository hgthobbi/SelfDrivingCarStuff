import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coords(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #print(parameters) # prints slope and y-int
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append([slope, intercept])
        else:
            right_fit.append([slope, intercept])
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coords(image, left_fit_average)
    right_line = make_coords(image, right_fit_average)
    return np.array([left_line, right_line])


def canny(image):
    # for a conversion from rgb to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # will be using gaussian filters to smoothen out the image
    # reducing noise in grayscale image, performed using a kernel (5, 5), with deviation of 0
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # using derivative to compute the change in the gradient of pixels using the Canny function
    canny = cv2.Canny(blur, 50, 150)
    # white lines are edges with sharp changes in intensity - most rapid changes in brightness
    # small changes are not shown at all, meaning the black parts of the image
    # specify a region of interest
    return canny

def displayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #print(line) # prints each line, each is 2D array with line coords in form [[x1, y1, x2, y2]] has one row and 4 columns
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # arg1: image to be drawn on, arg2 and arg3: 2 coords for line, color of line to be drawn, line thickness 
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255) # mask the image with a white triangle where we expect the lanes
    masked_image = cv2.bitwise_and(image, mask) 
    # this bitwise & will take the bitwise & of each pixel in the two arrays and will mask the canny image to only 
    # show the region of interest traced by a contour of the mask
    return masked_image

# image = cv2.imread('test_image.jpg')
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# # use these two to get a image with coordinates shown to estimate the region for lane line detection
# #plt.imshow(canny)
# #plt.show()
# cropped_image = region_of_interest(canny_image)
# # arg 1: image to apply the hough transform, arg 2: rho for height size of hough bin (y), arg 3: theta size for width size of bin (x)
# # fourth arguement is threshold -> min number of votes needed to accept a candidate line
# # fifth arg: placeholder array needed to pass; 6th arg: length of line in pixels we will accept into the output; any lines traced less than 40 will be rejected
# # 7th arg: max distance between lines which we will allow to be connected into single lines rather than splitting into separate lines
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = displayLines(lane_image, averaged_lines)
# # arg1: src1; arg2: weight of first array of elements; arg3: src2, weight of second array; output array; gamma -> scalar added to each sum
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap= 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = displayLines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("result", combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()








