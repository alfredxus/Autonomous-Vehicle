import cv2 as cv2
import numpy as np
import depthai as dai

def show_image(name, img):  # function for displaying the image
    cv2.imshow(name, img)
    cv2.waitKey(33)
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
    # bounds = np.array([[[120,185], [120, 320], [530, 320], [530, 185]]], dtype=np.int32)
    # bounds = np.array([[[0,200], [0, 400], [650, 400], [650, 200]]], dtype=np.int32)
    bounds = np.array([[[150, 120], [150, 280], [500, 280], [500, 120]]], dtype=np.int32)

    # bounds = np.array([[[0,image.shape[0]],[0,image.shape[0]/2],[900,image.shape[0]/2],[900,image.shape[0]]]],dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, bounds, [255, 255, 255])
    # show_image('inputmask',mask)
    masked_image = cv2.bitwise_and(image, mask)
    #show_image('mask', masked_image)
    return masked_image
def crop(image):
    return image[120:280,150:500,:]


def draw_lines(img, lines):  # function for drawing lines on black mask
    mask_lines = np.zeros_like(img)
    print("lines are",lines)

    for points in lines:
        print(points)
        # if points==None:
        #     print("suprise modafaka")
        x1, y1, x2, y2 = points[0]
        if x1>650 or x1<0:
            x1=325
        if y1>650 or y1<0:
            y1=0
        if x2>650 or x2<0:
            x2=650
        if y2>650 or y2<0:
            y2=0
        print(x1,y1,x2,y2)
        cv2.line(mask_lines, (x1, y1), (x2, y2), [0, 0, 255], 2)

    return mask_lines



def get_coordinates(img, line_parameters):  # functions for getting final coordinates
    slope = line_parameters[0]
    intercept = line_parameters[1]
    y1 =130
    y2 =10
    # y1=img.shape[0]
    # y2 = 0.6*img.shape[0]
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [x1, int(y1), x2, int(y2)]


def compute_average_lines(img, lines):
    left_lane_lines = []
    right_lane_lines = []
    left_weights = []
    right_weights = []
    if lines is None:
        return [[None], [None]]
    #print("lines",lines)
    for points in lines:
        x1, y1, x2, y2 = points[0]
        if lines is None:
            return None
        if x2 == x1:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # implementing polyfit to identify slope and intercept
        slope, intercept = parameters
        #print('param',parameters)
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < 0:
            left_lane_lines.append([slope, intercept])
            left_weights.append(length)
        else:
            right_lane_lines.append([slope, intercept])
            right_weights.append(length)
    # Computing average slope and intercept
    #print('left_lane',left_lane_lines)
    if np.size(left_lane_lines) ==0:
        left_average_line =None
        print("leftlanecomp")
        right_average_line = np.average(right_lane_lines, axis=0)
    elif np.size(right_lane_lines) ==0:
        left_average_line = np.average(left_lane_lines, axis=0)
        print("rightlanecomp")
        right_average_line = None
    else:
        left_average_line = np.average(left_lane_lines, axis=0)
        right_average_line = np.average(right_lane_lines, axis=0)
    #print('left average',left_average_line,'right average',right_average_line)

    if left_average_line is not None and right_average_line is not None:
        print('bug')
        left_fit_points = get_coordinates(img, left_average_line)
        right_fit_points = get_coordinates(img, right_average_line)
        # print(left_fit_points,right_fit_points)
        return [[left_fit_points], [right_fit_points]]  # returning the final coordinates

    elif right_average_line is not None:
        print("leftbug")
        right_fit_points = get_coordinates(img, right_average_line)
        return [[right_fit_points], [right_fit_points]]

    elif left_average_line is not None:
        print("rightbug")
        left_fit_points = get_coordinates(img, left_average_line)
        return[[left_fit_points], [left_fit_points]]
    # #Computing weigthed sum
    # if len(left_weights)>0:
    #     left_average_line = np.dot(left_weights,left_lane_lines)/np.sum(left_weights)
    # if len(right_weights)>0:
    #     right_average_line = np.dot(right_weights,right_lane_lines)/np.sum(right_weights)
    left_fit_points = get_coordinates(img, left_average_line)
    right_fit_points = get_coordinates(img, right_average_line)
    return[[left_fit_points],[right_fit_points]]

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
    return img

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
                print("red")
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
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
                print("green")
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

        # TODO: Break here


    if y_circles is not None:
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
                print()
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    # cv2.imshow('detected results', cimg)
    #cv2.imwrite(path+'//result//'+file, cimg)
    # cv2.imshow('maskr', maskr)
    # cv2.imshow('maskg', maskg)
    # cv2.imshow('masky', masky)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cimg


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

        #   TODO CROP frame
        # lane_image = region_of_interest(lane_image)
        # lane_image= crop(lane_image)
        lane_image =crop(lane_image)
        show_vid('region of interest',lane_image)
        lane_imagedet = detect(lane_image)
        show_vid('detection red/green', lane_imagedet)
        # lane_imaget=np.copy(lane_image)
        # lane_image2=only_white(lane_imaget)
        # show_vid('original(without yellow)', lane_image2)
        # lane_gray = cv2.cvtColor(lane_image2, cv2.COLOR_BGR2GRAY)
        # show_vid('gray',lane_gray)
        # lane_canny = find_canny(lane_image2, 100, 200)
        # show_vid('canny', lane_canny)
        # lane_roi = region_of_interest(lane_canny)
        #show_vid('roi',lane_roi)
        #lane_crop = crop(lane_canny)
        #show_vid('crop',lane_crop)
        # lane_lines = cv2.HoughLinesP(lane_canny,1,np.pi/180,50,40,5)
        # #print("laneline",lane_lines)
        # lane_lines_plotted = draw_lines(lane_image2, lane_lines)
        # # show_image('lines',lane_lines_plotted)
        # result_lines = compute_average_lines(lane_image2, lane_lines)
        # #print("result line",result_lines)
        # final_lines_mask = draw_lines(lane_image2, result_lines)
        # # show_image('final',final_lines_mask)
        # for points in result_lines:
        #     print("points",points)
        #     # if points==None:
        #     #     print("suprise modafaka")
        #     x1, y1, x2, y2 = points[0]
        #     if x1>650 or x1<0:
        #         x1=325
        #
        #     if y1>650 or y1<0:
        #         y1=0
        #     if x2>650 or x2<0:
        #         x2=650
        #     if y2>650 or y2<0:
        #         y2=0
        #     cv2.line(lane_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # show_vid('output',lane_image)