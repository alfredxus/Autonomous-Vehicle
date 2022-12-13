import time
import cv2 as cv2
import numpy as np
import depthai as dai

class VESC:
    def __init__(self, serial_port, percent=.2, has_sensor=False, start_heartbeat=True, baudrate=115200, timeout=0.05, steering_scale = 1.0, steering_offset = 0.0 ):
        
        try:
            import pyvesc
        except Exception as err:
            print("\n\n\n\n", err, "\n")
            print("please use the following command to import pyvesc so that you can also set")
            print("the servo position:")
            print("pip install git+https://github.com/LiamBindle/PyVESC.git@master")
            print("\n\n\n")
            time.sleep(1)
            raise
        
        assert percent <= 1 and percent >= -1,'\n\nOnly percentages are allowed for MAX_VESC_SPEED (we recommend a value of about .2) (negative values flip direction of motor)'
        self.steering_scale = steering_scale
        self.steering_offset = steering_offset
        self.percent = percent
        
        try:
            self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)
        except Exception as err:
            print("\n\n\n\n", err)
            print("\n\nto fix permission denied errors, try running the following command:")
            print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")
            time.sleep(1)
            raise
        
    def run(self, angle, throttle):
        self.v.set_servo((angle * self.steering_scale) + self.steering_offset)
        self.v.set_duty_cycle(throttle*self.percent)

def show_vid(name , img):
    cv2.imshow(name, img)
    cv2.waitKey(1)

def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    return img_canny

def region_of_interest(image):  # function for extracting region of interest
    # bounds in (x,y) format
    bounds = np.array([[[0, 400], [0, 380], [160, 310], [490, 310], [650, 380], [650, 400]]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

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

    if leftSlope and rightSlope and leftSlope > 0.15 and rightSlope > 0.15:
        print("Go straight")
        Vesc_object.run(0.5,0.2)
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
        print("Stop line ahead, stopping soon")
        goStraight()
        if stopLineDetected == False:
            lineInProgress(True)
            lineDetected(True)
        return [[0,0,0,0]], [stop_points]

    elif rightSlope and rightSlope < 0.05:
        stop_points = get_coordinates_horizontal(img, right_average_line)
        print("Stop line ahead, stopping soon")
        goStraight()
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
    return [[left_fit_points], [right_fit_points]]  # returning the final coordinates

def only_blue(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Mask the hsv image to find white
    lower_white = (90, 50, 70)
    upper_white = (128, 255, 255)
    mask = cv2.inRange(hsv_img, lower_white, upper_white)

    # Almost working:
    for channel in range(3):
        img[:, :, channel] = np.bitwise_and(mask, img[:, :, channel])
    return img,hsv_img

def detect(img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    #img = cv2.imread(img)
    cimg = img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    # lower_yellow = np.array([15,100,100])
    # upper_yellow = np.array([35,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)

    size = img.shape
    # print size

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)

    # traffic light detect
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:

        print("Red Circle")

        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                print("Red")
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:

        print("Green Circle")

        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                print("Green")
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

                setPendingTrafficResponse(False)
                setResumeTraffic(True)
                Vesc_object.run(0.5505, 0.2)
                time.sleep(5)
                Vesc_object.run(0.5505, 0)

    if y_circles is not None:

        print("Yellow Circle")

        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                print("Yellow")
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    return cimg

# Global variables
stopLineDetected = False
stopLineInProgress = False
lineInProgressCount = 0

pendingTrafficResponse = False
resumeTraffic = False

def lineDetected(boo):
    global stopLineDetected
    stopLineDetected = boo
    
def lineInProgress(boo):
    global stopLineInProgress
    global lineInProgressCount
    stopLineInProgress = boo
    if boo == True:
        lineInProgressCount += 1

def lineDetected(boo):
    global stopLineDetected
    stopLineDetected = boo

def goStraight():
    global lineInProgressCount
    if lineInProgressCount < 10:
        Vesc_object.run(0.5505, 0.15)
    elif lineInProgressCount < 20:
        Vesc_object.run(0.5505, 0.14)
    elif lineInProgressCount < 30:
        Vesc_object.run(0.5505, 0.13)
    elif lineInProgressCount < 40:
        Vesc_object.run(0.5505, 0.12)
    elif lineInProgressCount < 40:
        Vesc_object.run(0.5505, 0.11)

def setPendingTrafficResponse(boo):
    global pendingTrafficResponse
    pendingTrafficResponse = boo

def setResumeTraffic(boo):
    global resumeTraffic
    resumeTraffic = boo

# Control logic below
Vesc_object = VESC('/dev/ttyACM0')

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

        if pendingTrafficResponse:
            print("Pending traffic response...")
            traffic_image = lane_image[120:280,150:500,:]
            traffic_image_detected = detect(lane_image)
            time.sleep(0.5)
            continue

        if not resumeTraffic:
            lane_image = cv2.resize(lane_image,[650,400])
            lane_imaget = np.copy(lane_image)
            lane_image2,hsv_lane = only_blue(lane_imaget)
            lane_gray = cv2.cvtColor(lane_image2, cv2.COLOR_BGR2GRAY)
            lane_canny = find_canny(lane_image2, 50, 200)
            lane_roi = region_of_interest(lane_canny)
            lane_lines = cv2.HoughLinesP(lane_roi,1,np.pi/180,20,20,2)

            try:
                len(lane_lines)
            except:
                continue

            lane_image3 = lane_canny.copy()
            lane_lines_plotted = draw_lines(lane_imaget, lane_lines)
            result_lines = compute_average_lines(lane_image2, lane_lines)

        if stopLineInProgress == True and stopLineDetected == False:
            print("Stop now")
            time.sleep(2.5)
            Vesc_object.run(0.5505,0)
            lineInProgress(False)
            setPendingTrafficResponse(True)
