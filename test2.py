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

def find_canny(img, thresh_low, thresh_high):  # function for implementing the canny
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, thresh_low, thresh_high)
    return img_canny

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
        Vesc_object.run(0.5,0.2)
    elif leftSlope and rightSlope and leftSlope < 0.3 and rightSlope < 0.3:
        if leftSlope > rightSlope:
            print("Turn right")
            Vesc_object.run(0.7,0.2)
        else:
            print("Turn left")
            Vesc_object.run(0.3,0.2)
    elif leftSlope and leftSlope < 0.3:
        print("Turn right")
        Vesc_object.run(0.7,0.2)
    elif rightSlope and rightSlope < 0.3:
        print("Turn left")
        Vesc_object.run(0.3,0.2)

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
