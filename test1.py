import cv2 as cv2
import numpy as np
import depthai as dai

def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_vid(name , img):
    cv2.imshow(name, img)
    cv2.waitKey(1)

def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show_image('gray',img_gray)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # show_image('blur',img_blur)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    #show_image('Canny', img_canny)
    return img_canny

def region_of_interest(image):  # function for extracting region of interest
    # bounds in (x,y) format
    # bounds = np.array([[[0, 250], [0, 200], [150, 100], [500, 100], [650, 200], [650, 250]]], dtype=np.int32)
    # bounds = np.array([[[0, 400], [0, 350], [150, 250], [500, 250], [650, 250], [650, 400]]], dtype=np.int32)
    bounds = np.array([[[0, 400], [0, 350], [150, 650], [500, 650], [650, 250], [650, 400]]], dtype=np.int32)

    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image, mask)
    # show_image('mask', masked_image)
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
    # y1 = 130
    # y2 = 10
    y1 = 95
    y2 = 15
    # y1=img.shape[0]
    # y2 = 0.6*img.shape[0]
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
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
    # print('left average',left_average_line,'right average',right_average_line)

    leftSlope = None
    rightSlope = None

    if np.size(left_average_line) > 1:
        leftSlope = abs(left_average_line[0])
    if np.size(right_average_line) > 1:
        rightSlope = right_average_line[0]

    #print("Left Slope = " + leftSlope + ", Right Slope = " + rightSlope)
    
    if leftSlope and rightSlope and leftSlope > 0.3 and rightSlope > 0.3:
        print("Go straight")
    elif leftSlope and rightSlope and leftSlope < 0.3 and rightSlope < 0.3:
        if leftSlope > rightSlope:
            print("Turn right")
        else:
            print("Turn left")
    elif leftSlope and leftSlope < 0.3:
        print("Turn right")
    elif rightSlope and rightSlope < 0.3:
        print("Turn left")

    if np.size(left_average_line) == 1 and np.isnan(left_average_line):
        right_fit_points = get_coordinates(img, right_average_line)
        # return [[right_fit_points], [right_fit_points]]
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
    lower_white = (0, 0, 100)
    upper_white = (255, 30, 255)
    mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Almost working:
    for channel in range(3):
        img[:, :, channel] = np.bitwise_and(mask, img[:, :, channel])
    return img,hsv_img

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(650, 400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)
    # Bootloader version
    if device.getBootloaderVersion() is not None:
        print('Bootloader version: ', device.getBootloaderVersion())

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        frame = inRgb.getCvFrame()
        if frame is None:
            print(' No captured frame -- Break!')
            break
        lane_image = np.copy(frame)

        lane_image = cv2.resize(lane_image,[650,400])
        lane_image= crop(lane_image)
        lane_imaget=np.copy(lane_image)
        lane_image2,hsv_lane=only_white(lane_imaget)
        lane_gray = cv2.cvtColor(lane_image2, cv2.COLOR_BGR2GRAY)
        lane_canny = find_canny(lane_image2, 30, 120)
        lane_lines = cv2.HoughLinesP(lane_canny,1,np.pi/180,65,200,10)

        try:
            len(lane_lines)
        except:
            continue

        lane_image3 = lane_canny.copy()
        lane_lines_plotted = draw_lines(lane_imaget, lane_lines)
        result_lines = compute_average_lines(lane_image2, lane_lines)
        final_lines_mask = draw_lines(lane_image2, result_lines)
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

            cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
