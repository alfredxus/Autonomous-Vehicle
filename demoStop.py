import time
import cv2 as cv2
import numpy as np

def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_vid(name , img):
    cv2.imshow(name, img)
    cv2.waitKey(1)

def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)

    return img_canny

def region_of_interest(image):  # function for extracting region of interest
    bounds = np.array([[[0, 400], [0, 380], [160, 310], [490, 310], [650, 380], [650, 400]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def crop(image):
    return image[250:,:,:]

def draw_lines(img, lines):  # function for drawing lines on black mask
    mask_lines = np.zeros_like(img)
    for points in lines:
        x1, y1, x2, y2 = points[0]

        if x1 not in range(650) or y1 not in range(650) or x2 not in range(650) or y2 not in range(650):
            x1 = 0
            y1 = 0
            x2 = 650
            y2 = 0

        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)

    return mask_lines


def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    slope = line_parameters[0]

    if slope == float("inf") or slope == 0.0:
        return [0,0,0,0]

    intercept = line_parameters[1]
    y1 = 330
    y2 = 400
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [int(x1), int(y1), int(x2), int(y2)]

def get_coordinates_horizontal(img, line_parameters):  # functions for getting stop coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    x1 = 0
    x2 = 650
    y1 = abs(int(x1 * slope - intercept))
    y2 = abs(int(x2 * slope - intercept))
    return [int(x1), int(y1), int(x2), int(y2)]

def compute_average_lines(img, lines):
    left_lane_lines = []
    right_lane_lines = []
    left_weights = []
    right_weights = []
    for points in lines:
        x1, y1, x2, y2 = points[0]
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # implementing polyfit to identify slope and intercept
        slope, intercept = parameters
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < 0:
            left_lane_lines.append([slope, intercept])
            left_weights.append(length)
        else:
            right_lane_lines.append([slope, intercept])
            right_weights.append(length)

    # Computing average slope and intercept
    left_average_line = np.average(left_lane_lines, axis=0)
    right_average_line = np.average(right_lane_lines, axis=0)

    leftSlope = None
    rightSlope = None

    if np.size(left_average_line) > 1:
        leftSlope = abs(left_average_line[0])
    if np.size(right_average_line) > 1:
        rightSlope = right_average_line[0]

    lineDetected(False)
        
    # print("Left Slope = " + str(leftSlope) + ", Right Slope = " + str(rightSlope))
    if leftSlope and rightSlope and leftSlope > 0.15 and rightSlope > 0.15:
        print("Go straight")
    elif leftSlope and rightSlope and leftSlope < 0.15 and rightSlope < 0.15:
        if leftSlope > rightSlope:
            if leftSlope < 0.05:
                print("Stop line")
            else:
                print("Turn right (compare)")
        else:
            if rightSlope < 0.05:
                print("Stop line")
            else:
                print("Turn left (compare)")
    elif leftSlope and leftSlope < 0.05:
        stop_points = get_coordinates_horizontal(img, left_average_line)
        print("Stop line ahead, reducing speed")

        if stopLineDetected == False:
            lineInProgress(True)
            lineDetected(True)

        return [[0,0,0,0]], [stop_points]
    elif rightSlope and rightSlope < 0.05:
        stop_points = get_coordinates_horizontal(img, right_average_line)
        print("Stop line ahead, reducing speed")

        if stopLineDetected == False:
            lineInProgress(True)
            lineDetected(True)

        return [stop_points], [[0,0,0,0]]

    if np.size(left_average_line) == 1 and np.isnan(left_average_line):
        right_fit_points = get_coordinates(img, right_average_line)
        return [[[0,0,0,0]], [right_fit_points]]
    
    elif np.size(right_average_line) == 1 and np.isnan(right_average_line):
        left_fit_points = get_coordinates(img, left_average_line)
        return [[left_fit_points], [[0,0,0,0]]]
        
    elif np.size(left_average_line) == 1 and np.isnan(left_average_line) and np.size(right_average_line) == 1 and np.isnan(right_average_line):
        return [[0,0,0,0], [[0,0,0,0]]]

    left_fit_points = get_coordinates(img, left_average_line)
    right_fit_points = get_coordinates(img, right_average_line)
    # print(left_fit_points,right_fit_points)
    return [[left_fit_points], [right_fit_points]]  # returning the final coordinates

def remove_yellow(img): #WIP

    yellow = np.uint8([[[255, 255, 0]]])  # here insert the bgr values which you want to convert to hsv4
    hsvYellow = cv2.cvtColor(yellow, cv2.COLOR_BGR2HSV)
    lowerLimit = hsvYellow[0][0][0] - 10, 100, 100
    upperLimit = hsvYellow[0][0][0] + 10, 255, 255
    #print(lowerLimit,upperLimit)
    lower = np.array([80, 100, 100])  # -- Lower range --
    upper = np.array([100, 255, 255]) # -- Upper range --
    mask = cv2.inRange(img, lower, upper)
    res = cv2.bitwise_and(img, img, mask=mask)  # -- Contains pixels having the gray color--
    return res


color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}

def only_white(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask the hsv image to find white
    lower_white = (90, 50, 70)
    upper_white = (128, 255, 255)
    mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Almost working:
    for channel in range(3):
        img[:, :, channel] = np.bitwise_and(mask, img[:, :, channel])
    return img,hsv_img

# Global variables
stopLineDetected = False
stopLineInProgress = False
lineInProgressCount = 0

def lineDetected(boo):
    global stopLineDetected
    stopLineDetected = boo
    
def lineInProgress(boo):
    global stopLineInProgress
    global lineInProgressCount
    stopLineInProgress = boo
    if boo == True:
        lineInProgressCount += 1

def drawLine(img, pt1, pt2, programCount):
    if programCount < 1 or programCount >= 50:
        cv2.line(img, pt1, pt2, (0, 0, 255), 4)
    elif programCount < 10:
        cv2.line(img, pt1, pt2, (0, 255, 0), 4)
    elif programCount < 20:
        cv2.line(img, pt1, pt2, (0, 255, 128), 4)
    elif programCount < 30:
        cv2.line(img, pt1, pt2, (0, 255, 255), 4)
    else:
        cv2.line(img, pt1, pt2, (0, 128, 255), 4)

cap = cv2.VideoCapture('intersection.mp4')
if not cap.isOpened:
    print('Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print(' No captured frame -- Break!')
        break
    lane_image = np.copy(frame)

    lane_image = cv2.resize(lane_image,[650,400])
    # lane_image= crop(lane_image)
    lane_imaget=np.copy(lane_image)
    lane_image2,hsv_lane=only_white(lane_imaget)
    #show_image('HSV',hsv_lane)
    #import matplotlib.pyplot as plt
    #plt.imshow(hsv_lane)
    #plt.show()
    show_vid('HSV', hsv_lane)
    lane_gray = cv2.cvtColor(lane_image2, cv2.COLOR_BGR2GRAY)
    # show_vid('gray',lane_gray)
    # lane_canny = find_canny(lane_image2, 35, 115)
    lane_canny = find_canny(lane_image2, 50, 200)
    show_vid('canny', lane_canny)
    # lane_roi = region_of_interest(lane_canny)
    lane_roi = region_of_interest(lane_canny)
    # show_vid('roi',lane_roi)
    #lane_crop = crop(lane_canny)
    #show_vid('crop',lane_crop)
    # lane_lines = cv2.HoughLinesP(lane_canny,1,np.pi/180,80,180,6)
    lane_lines = cv2.HoughLinesP(lane_roi,1,np.pi/180,20,20,2)

    try:
        len(lane_lines)
    except:
        continue

    # print("laneline",lane_lines)
    lane_image3 = lane_canny.copy()
    lane_lines_plotted = draw_lines(lane_imaget, lane_lines)
    # show_image('lines',lane_lines_plotted)
    show_vid('intermediate',lane_lines_plotted)
    result_lines = compute_average_lines(lane_image2, lane_lines)
    # print("result line",result_lines)
    final_lines_mask = draw_lines(lane_image2, result_lines)
    # show_vid('intermediate',lane_image2)
    # show_image('final',final_lines_mask)


    # Motion control below
    if stopLineInProgress == True and stopLineDetected == False:
        print("Stop now")
        lineInProgress(False)
        time.sleep(2)
        break

    for points in result_lines:
        x1, y1, x2, y2 = points[0]
 
        if x1 < 0:
            x1 = 0
        elif x1 > 650:
            x1 = 650

        if y1 < 0:
            y1 = 0
        elif y1 > 400:
            y1 = 400

        if x2 < 0:
            x2 = 0
        elif x2 > 650:
            x2 = 650

        if y2 < 0:
            y2 = 0
        elif y2 > 400:
            y2 = 400
        
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue

        # cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
        drawLine(lane_image, (x1, y1), (x2, y2), lineInProgressCount)

    show_vid('output',lane_image)
    # show_vid('output',lane_image2)
