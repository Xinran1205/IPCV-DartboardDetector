import cv2
import numpy as np
import os
import sys
import argparse
import copy
import time

# runing example : python task3.py -names Dartboard/dart0.jpg
# default running all the images in the Dartboard folder
totalF1 = 0


# this function receives an image and doing the hough line transform
# it returns the accumulator list, thetas list and rhos list
def hough_line_transform(image):

    # thetas[0] = -90 degree, thetas[1] = -89 degree, thetas[179] = 89 degree
    thetas = np.deg2rad(np.arange(-90, 90))
    width, height = image.shape
    # calculate the diagonal length of the image
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # Max possible rho value
    # this function call generates an array containing 2*diag_len numbers from -diag_len to diag_len
    # same as thetas, rhos[0] is the value of -diag_len
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)

    # Cache some reusable values
    # cos_arr, an array containing the cosine values of each angle in thetas.
    cos_arr = np.cos(thetas)
    sin_arr = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize the accumulator space to zeros
    # generate an array with 2*diag_len rows and num_thetas columns, each element is 0.
    # used to store the votes of each pair of parameters
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)

    # Find edge points (pixels with value 1)
    # this step is very good, it can directly find all the edge points in the image,
    # and then store the coordinates of these points in x_idxs and y_idxs.
    # save the time of traversing all pixels.
    y_idxs, x_idxs = np.nonzero(image)
    # x_idxs and y_idxs are of the same length, they are one-to-one pairs,
    # each pair represents the coordinates of an edge point.
    # Loop through edge points and populate the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # In the Hough transform, ρ can be negative or positive,
            # representing the directed distance from the line to the image origin.
            # In order to use this value as an index in the array,
            # we need to offset it by a length of diag_len to ensure that the index is positive.
            # This is because the index of the array cannot be negative.
            # Therefore, the calculated ρ value plus diag_len becomes a non-negative integer,
            # which can be used as an index of the accumulator array accumulator.
            # This is very clever. For example, if diag_len is -120, then there are 240 numbers in rhos,
            # indexes 0-239, and each number rho takes a value of -120 to 120.
            # Then I add 120 to rho, and the value of rho becomes 0 to 240, which corresponds to indexes 0-240.
            rho = int(round(x * cos_arr[t_idx] + y * sin_arr[t_idx]) + diag_len)

            # be careful, the rho and t_idx in accumulator are just indexes, not the real rho and t_idx values.
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos

def is_inside(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    return x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    non_max = np.zeros((M, N), dtype=np.float32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q, r = 255, 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    non_max[i, j] = gradient_magnitude[i, j]
                else:
                    non_max[i, j] = 0
            except IndexError:
                pass
    return non_max


def double_thresholding(img, low_ratio=0.5, high_ratio=0.06):
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong, weak = 255, 0

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result


def sobelEdgeDetect(img, img_name):
    # 1. Convert the image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Blur the image using a Gaussian filter
    AfterBlur = cv2.GaussianBlur(img, (7, 7), 0)

    # 3. calculate x direction derivative
    DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
    # cv2.imwrite("DerivativeX.jpg", np.clip(DerivativeX + 128, 0, 255))

    # 4. calculate y direction derivative
    DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
    # cv2.imwrite("DerivativeY.jpg", np.clip(DerivativeY + 128, 0, 255))

    # 6.
    Direction = np.arctan2(DerivativeY, DerivativeX)

    # 5. calculate gradient magnitude
    Magnitude = np.sqrt(DerivativeX ** 2 + DerivativeY ** 2)
    normalized_magnitude = (Magnitude / np.max(Magnitude) * 255).astype(np.uint8)
    normalized_magnitude = non_maximum_suppression(normalized_magnitude, Direction)
    normalized_magnitude = double_thresholding(normalized_magnitude)
    cv2.imwrite(f"task3output/magnitude/Magnitude_{img_name}.jpg", normalized_magnitude)


def threshold_image(image, threshold):
    _, threshed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshed


def houghTransformation(thresholdImg, DirectionImg, min_radius, max_radius):
    width = thresholdImg.shape[1]
    height = thresholdImg.shape[0]
    hough_space = np.zeros((height, width, max_radius + 1))
    for y in range(height):
        for x in range(width):
            if thresholdImg[y, x] == 255:
                for r in range(min_radius, max_radius + 1):
                    # for a circle, the middle point is (a, b) and it on the gradiant line of (x, y)
                    a = int(x - r * np.cos(DirectionImg[y, x]))
                    b = int(y - r * np.sin(DirectionImg[y, x]))
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[b, a, r] += 1
                    a = int(x + r * np.cos(DirectionImg[y, x]))
                    b = int(y + r * np.sin(DirectionImg[y, x]))
                    if a >= 0 and a < width and b >= 0 and b < height:
                        hough_space[b, a, r] += 1
    return hough_space


def find_parameter(hough_space, thresholdH):
    width = hough_space.shape[1]
    height = hough_space.shape[0]
    radius = hough_space.shape[2]
    parameter = []
    for y in range(height):
        for x in range(width):
            for r in range(radius):
                if hough_space[y, x, r] >= thresholdH:
                    parameter.append([y, x, r])
    return parameter

def draw_circle(image, parameters):
    parameters = filter_similar_circles(parameters)
    for param in parameters:
        cv2.circle(image, (param[1], param[0]), param[2], (0, 255, 255), 2)
    return image


def filter_similar_circles(parameters, result1_rectangle_for_circle, iou_threshold=0.1):
    # Sort circles by radius in descending order
    parameters.sort(key=lambda x: x[2], reverse=True)

    filtered_params = []

    for i, current_circle in enumerate(parameters):
        is_similar = False
        current_box = result1_rectangle_for_circle[i]

        for j, saved_circle in enumerate(filtered_params):
            saved_box = result1_rectangle_for_circle[filtered_params.index(saved_circle)]

            # Calculate IOU between current circle box and saved circle box
            iou = computeIoU(current_box, saved_box)

            center_distance = ((current_circle[0] - saved_circle[0]) ** 2 +
                               (current_circle[1] - saved_circle[1]) ** 2) ** 0.5
            radius_diff = abs(current_circle[2] - saved_circle[2])

            # Check IOU and also ensure that the smaller circle is not completely inside the larger one
            if iou > iou_threshold or (center_distance < radius_diff+5 and current_circle[2] < saved_circle[2]+5):
                is_similar = True
                break

        if not is_similar:
            filtered_params.append(current_circle)

    return filtered_params


# this is the new function for task 3!!!!!!
# this function combined the hough circle transform and the hough line transform
# line_threshold represents how many points need to be on a line for it to be considered a line.
def detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold):
    # find the possible circles
    potential_circles = find_parameter(hough_space, thresholdH)

    dartboards = []
    # apply hough line transform to this area(viola detected area)
    accumulator, thetas, rhos = hough_line_transform(thresholdImg)
    # This line_threshold represents how many points need to be on a line for it to be considered a line.
    idxs = np.where(accumulator > line_threshold)
    # for each circle in this area, check how many lines passing through
    for circle in potential_circles:
        converging_lines = 0
        for i in range(len(idxs[0])):
            rho = rhos[idxs[0][i]]
            theta = thetas[idxs[1][i]]
            # check if the vertex is on the line
            # the polar equation of the line is rho = x*cos(theta) + y*sin(theta)
            # if the vertex satisfies this equation, it is on the line
            # 0.5 is a falut tolerance
            if abs(rho - (circle[1] * np.cos(theta) + circle[0] * np.sin(theta))) < 0.5:
                converging_lines += 1
        # this 80 is a threshold for how many lines need to be converging to the center to be considered a dartboard
        if converging_lines > 80:
            dartboards.append(circle)

    return dartboards

# this is the crucial function of task 3,
# this function combined the viola-jones detection and hough transformation(circle and line)
# thresholdH is the threshold of the hough circle transform
# line_threshold is the threshold of the hough line transform
def detect_dartboards_with_hough(frame, model, img_name, scaleFactor=1.008, minNeighbors=25, flags=0, minSize=(50, 50),
                                 maxSize=(300, 300), thresholdH=6,line_threshold=40):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Viola-Jones detection
    VioJonResult = model.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, flags=flags,
                                          minSize=minSize, maxSize=maxSize).tolist()
    VioJonResult = non_maximum_suppressionForViola(VioJonResult, 0.2)

    print(f"Detected {len(VioJonResult)} VioJonResult")

    # draw the groundtruth on the image
    current_image_name = os.path.splitext(image_file.split('/')[-1])[0]
    groundtruth = readGroundtruth()
    for img_name_truth in groundtruth:
        if img_name_truth == current_image_name:
            for bbox in groundtruth[img_name_truth]:
                start_point = (bbox[0], bbox[1])
                end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                colour = (0, 0, 255)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    # get the preprocessing image for hough transformation
    AfterBlur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
    DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
    Direction = np.arctan2(DerivativeY, DerivativeX)
    cv2.imwrite(f"task3output/threshold/threshold_{img_name}.jpg",
                threshold_image(cv2.imread(f"task3output/magnitude/Magnitude_{img_name}.jpg", cv2.IMREAD_GRAYSCALE),
                                50))

    thresholdImgWhole = cv2.imread(f"task3output/threshold/threshold_{img_name}.jpg", cv2.IMREAD_GRAYSCALE)
    total_dartboard_circles = []
    # expand size, I want to do the hough transformation in the area that is slightly larger than the viola-jones box
    expand_size = 16
    # for each viola-jones detected box, do the hough transformation to find the circle
    for (x, y, w, h) in VioJonResult:
        # calculate the new coordinate of the cropped image, make sure it is not out of the image
        x_start = max(x - expand_size, 0)
        y_start = max(y - expand_size, 0)
        x_end = min(x + w + expand_size, frame_gray.shape[1])
        y_end = min(y + h + expand_size, frame_gray.shape[0])
        # create the cropped threshold image and direction image
        thresholdImg = thresholdImgWhole[y_start:y_end, x_start:x_end]
        Direction_cropped = Direction[y_start:y_end, x_start:x_end]

        # apply hough circle transformation on the cropped image
        hough_space = houghTransformation(thresholdImg, Direction_cropped, 10, 200)

        # use the hough line transformation to double check if it is a dartboard
        dartboard_circles = detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold)

        for board in dartboard_circles:
            # adjust the coordinate of the circle to the global coordinate
            total_dartboard_circles.append((board[0] + y_start, board[1] + x_start, board[2]))

    circle1 = []
    rectangle_for_circle = []
    rectangle_for_circle_copy = []
    for board in total_dartboard_circles:
        # for each circle, build a rectangle bounding box for it
        # in order to calculate the IOU
        top_left_x = board[1] - board[2]
        top_left_y = board[0] - board[2]
        width_height = 2 * board[2]
        rectangle_for_circle.append((top_left_x, top_left_y, width_height, width_height))

    # Filter 1:
    # for each viola-jones detected box, find the circle that has the largest IOU with it
    for Board in VioJonResult:
        best_iou = 0
        best_pred_index = -1
        for i, pred in enumerate(rectangle_for_circle):
            iou = computeIoU(Board, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_index = i
        if best_iou > 0.3 :
            # can be understood as these two indexes are shared
            rectangle_for_circle_copy.append(rectangle_for_circle[best_pred_index])
            circle1.append(total_dartboard_circles[best_pred_index])


    # Filter 2:
    # here I filter the circle again, based on the IOU and the rule that a big circle cannot contain a small circle
    result_circle = filter_similar_circles(circle1,rectangle_for_circle_copy)

    # create the final rectangle bounding boxes for the rest of circle
    result_rectangle_for_circle = []
    for board in result_circle:
        cv2.circle(frame, (board[1], board[0]), board[2], (0, 255, 0), 2)
        top_left_x = board[1] - board[2]
        top_left_y = board[0] - board[2]
        width_height = 2 * board[2]
        result_rectangle_for_circle.append((top_left_x, top_left_y, width_height, width_height))

    TPR, F1 = evaluate_predictions(result_rectangle_for_circle, groundtruth[current_image_name])
    print("TPR: ", TPR)
    print("F1: ", F1)
    global totalF1
    totalF1 += F1
    with open(f"task3output/evaluation_results.txt", 'a') as file:
        file.write(f"TPR: {TPR}\n")
        file.write(f"F1: {F1}\n")
    return frame


def non_maximum_suppressionForViola(boxes, iou_threshold):

    boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)
    final_boxes = []

    for current_box in boxes:

        if not final_boxes:
            final_boxes.append(current_box)
            continue

        keep_current_box = True
        for final_box in final_boxes:
            if is_inside(current_box, final_box):
                keep_current_box = False
                break

            if computeIoU(current_box, final_box) > iou_threshold:
                keep_current_box = False
                break

        if keep_current_box:
            final_boxes.append(current_box)

    return final_boxes


def computeIoU(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    # XA YA is the top left corner of the intersection rectangle
    # XB YB is the bottom right corner of the intersection rectangle
    # Compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute the Intersection over Union
    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

def evaluate_predictions(predictions, ground_truths, iou_threshold=0.15):
    predictions_copy = copy.deepcopy(predictions)

    TP = 0
    FP = 0
    FN = 0

    # For each ground truth, find the best matching prediction
    for gt in ground_truths:
        best_iou = 0
        best_pred_index = -1
        isInside = False
        for i, pred in enumerate(predictions_copy):
            if is_inside(gt, pred):
                isInside = True
                best_pred_index = i
                break
            iou = computeIoU(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_index = i

        # If the best matching prediction has IOU > threshold, it's a TP. Otherwise, it's a FN.
        if best_iou > iou_threshold or isInside:
            TP += 1
            # Remove this prediction from further consideration
            predictions_copy.pop(best_pred_index)
        else:
            FN += 1

    # Any remaining predictions are FP
    FP = len(predictions_copy)

    # Calculate TPR (true positive rate)
    TPR = TP / (TP + FN)

    # Calculate precision and recall
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TPR

    # Calculate F1 score
    F1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return TPR, F1

def readGroundtruth(filename='groundtruth.txt'):
    groundtruth = {}
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            img_name = content_list[0]
            x = int(float(content_list[1]))
            y = int(float(content_list[2]))
            width = int(float(content_list[3]))
            height = int(float(content_list[4]))

            bbox = (x, y, width, height)

            if img_name in groundtruth:
                groundtruth[img_name].append(bbox)
            else:
                groundtruth[img_name] = [bbox]

    return groundtruth


start_time = time.time()
if os.path.exists("task3output/evaluation_results.txt"):
    # If it does, delete the file
    os.remove("task3output/evaluation_results.txt")
#
parser = argparse.ArgumentParser(description='dart detection')
parser.add_argument('-names', nargs='+', default=['Dartboard/dart0.jpg','Dartboard/dart1.jpg','Dartboard/dart2.jpg',
                                                'Dartboard/dart3.jpg','Dartboard/dart4.jpg','Dartboard/dart5.jpg',
                                                'Dartboard/dart6.jpg','Dartboard/dart7.jpg','Dartboard/dart8.jpg',
                                                'Dartboard/dart9.jpg','Dartboard/dart10.jpg','Dartboard/dart11.jpg',
                                                'Dartboard/dart12.jpg','Dartboard/dart13.jpg','Dartboard/dart14.jpg',
                                                'Dartboard/dart15.jpg'])
args = parser.parse_args()

# image_files = [f"Dartboard/dart{i}.jpg" for i in range(16)]

for image_file in args.names:

    image_name = os.path.splitext(os.path.basename(image_file))[0]

    if not os.path.isfile(image_file):
        print(f'No such file: {image_file}')
        continue

    frame = cv2.imread(image_file, 1)
    if not (type(frame) is np.ndarray):
        print(f'Not image data: {image_file}')
        continue

    # load the Viola-Jones model
    cascade_name = "Dartboardcascade/cascade.xml"
    model = cv2.CascadeClassifier()
    if not model.load(cascade_name):
        print('--(!)Error loading cascade model')
        continue

    # doing edge detection and save the result
    sobelEdgeDetect(frame, image_name)

    # detect dartboards using Viola-Jones and Hough(circle and line) Transform
    result_image = detect_dartboards_with_hough(frame, model, image_name)

    # save the result image
    cv2.imwrite(f"task3output/detected_{image_name}.jpg", result_image)

with open("task3output/evaluation_results.txt", 'a') as file:
    averageF1 = totalF1 / len(args.names)
    file.write(f"AvgF1: {averageF1}\n")

end_time = time.time()
runtime = end_time - start_time
with open("task3output/evaluation_results.txt", 'a') as file:
    file.write(f"Runtime: {runtime} seconds\n")