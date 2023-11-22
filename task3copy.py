import cv2
import numpy as np
import os
import sys
import argparse
import copy


# 这个里面没有霍夫直线检测

def hough_line_transform(image):
    # Define the Hough space
    # np.arange(-90, 90)：这个函数调用生成了一个数组，包含从-90到89的整数。
    # np.deg2rad()：这个函数调用将角度转换为弧度。

    # 所以thetas是一个数组，包含了从-90度到89度的弧度值。thetas[0]是-90度的弧度值及-1.57
    thetas = np.deg2rad(np.arange(-90, 90))
    width, height = image.shape
    # 计算对角线的长度
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # Max possible rho value
    # 这个函数调用生成了一个数组，包含了从-diag_len到diag_len的2*diag_len个数。
    # 和前面的thetas差不多，rhos[0]是-diag_len的值。
    rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)

    # Cache some reusable values
    # cos_arr,一个数组，包含了thetas中每个角度的余弦值。
    cos_arr = np.cos(thetas)
    sin_arr = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize the accumulator space to zeros
    # 生成一个2*diag_len行，num_thetas列的数组，每个元素都是0。用来存放每一对参数的票数
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)

    # Find edge points (pixels with value 1)
    # 这一步很好，可以直接找到图像中所有的边缘点，然后把这些点的坐标存放在x_idxs和y_idxs中。
    # 省去遍历所有像素点的时间。
    y_idxs, x_idxs = np.nonzero(image)
    # x_idxs和y_idxs长度相同，他们是一对一对的，每一对代表一个边缘点的坐标。
    # Loop through edge points and populate the accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # 为什么要加diag_len：在霍夫变换中，ρ可以是负值或正值，表示直线到图像原点的有向距离。为了能在数组中使用这个值作为索引，
            # 我们需要将其偏移一个diag_len的长度，确保索引是正数。这是因为数组的索引不能是负数。
            # 因此，计算得到的ρ值加上diag_len后，就变成了一个非负整数，可以用作累加器数组accumulator的索引。
            # 这里很巧妙，比如diag_len是-120，那rhos里面有240个数，索引0-239，每个数rho取值是-120到120，那么我给rho加上120后，
            # rho的取值就变成了0到240，正好对应索引0-240。
            rho = int(round(x * cos_arr[t_idx] + y * sin_arr[t_idx]) + diag_len)

            # 注意accumulator中的rho和t_idx都只是索引（正的），不是真正的rho和t_idx值。
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


# def hough_line_transform(image):
#     thetas = np.deg2rad(np.arange(-90, 90))
#     width, height = image.shape
#     diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
#     rhos = np.linspace(-diag_len, diag_len, 2 * diag_len)
#
#     cos_arr = np.cos(thetas)
#     sin_arr = np.sin(thetas)
#     num_thetas = len(thetas)
#
#     accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
#
#     y_idxs, x_idxs = np.nonzero(image)
#
#     # Vectorized calculation of accumulator values
#     pixel_vals = image[y_idxs, x_idxs]
#     rho_vals = np.round(np.outer(x_idxs, cos_arr) + np.outer(y_idxs, sin_arr)) + diag_len
#     rho_vals = rho_vals.astype(int)
#     np.add.at(accumulator, (rho_vals, np.arange(num_thetas)), pixel_vals)
#
#     return accumulator, thetas, rhos


# 这个里面没有霍夫直线检测

def is_inside(box1, box2):
    # 如果box1在box2内部，返回True
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


def double_thresholding(img, low_ratio=0.1, high_ratio=0.2):
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
    # normalized_magnitude = non_maximum_suppression(normalized_magnitude, Direction)
    # normalized_magnitude = double_thresholding(normalized_magnitude)
    cv2.imwrite(f"task3copy/magnitude/Magnitude_{img_name}.jpg", normalized_magnitude)


def threshold_image(image, threshold):
    _, threshed = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return threshed


def houghTransformation(thresholdImg, DirectionImg, min_radius, max_radius):
    width = thresholdImg.shape[1]
    height = thresholdImg.shape[0]
    # 因为圆心坐标一定在height和width范围内
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


# def filter_largest_circles(parameters, center_error=40, radius_error=10):
#     # 使用字典来按圆心分组圆
#     circles_by_center = {}
#     for y, x, r in parameters:
#         # 查找已有圆心是否存在
#         found = False
#         for (cy, cx), (cr, _) in circles_by_center.items():
#             if ((cy - y) ** 2 + (cx - x) ** 2) ** 0.5 <= center_error:
#                 found = True
#                 # 如果当前圆的半径比已记录的圆更大，则更新
#                 if r > cr and abs(r - cr) > radius_error:
#                     circles_by_center[(cy, cx)] = (r, (y, x, r))
#                 break
#         if not found:
#             circles_by_center[(y, x)] = (r, (y, x, r))
#
#     # 只保留每组中半径最大的圆
#     return [info[1] for info in circles_by_center.values()]


def filter_similar_circles(parameters, center_threshold=50, radius_threshold=120):
    # Sort circles by radius in descending order
    parameters.sort(key=lambda x: x[2], reverse=True)

    filtered_params = []

    for current_circle in parameters:
        is_similar = False
        for saved_circle in filtered_params:
            center_distance = ((current_circle[0] - saved_circle[0]) ** 2 +
                               (current_circle[1] - saved_circle[1]) ** 2) ** 0.5
            radius_diff = abs(current_circle[2] - saved_circle[2])

            if center_distance < center_threshold and radius_diff < radius_threshold:
                is_similar = True
                # No need to replace since we are already iterating from largest to smallest
                break

        if not is_similar:
            filtered_params.append(current_circle)

    return filtered_params


def draw_circle(image, parameters):
    parameters = filter_similar_circles(parameters)
    for param in parameters:
        cv2.circle(image, (param[1], param[0]), param[2], (0, 255, 255), 2)
    return image


# def detect_dartboard(hough_space, thresholdH):
#     # 找到可能的圆
#     potential_circles = find_parameter(hough_space, thresholdH)
#     # 过滤相似的圆
#     # potential_circles = filter_similar_circles(potential_circles)
#     # potential_circles = filter_largest_circles(potential_circles)
#
#     return potential_circles

def detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold):
    # 找到可能的圆
    potential_circles = find_parameter(hough_space, thresholdH)
    # 过滤相似的圆

    # 确定飞镖盘
    dartboards = []
    # 针对这块区域（viola检测出的区域）计算直线的霍夫变换
    accumulator, thetas, rhos = hough_line_transform(thresholdImg)
    for circle in potential_circles:
        # 这个阈值代表多少才算一条线
        idxs = np.where(accumulator > line_threshold)
        converging_lines = 0
        for i in range(len(idxs[0])):
            rho = rhos[idxs[0][i]]
            theta = thetas[idxs[1][i]]
            # 检查顶点是否在直线上
            # 直线的极坐标方程是 rho = x*cos(theta) + y*sin(theta)
            # 如果顶点满足这个方程，它就在直线上
            if abs(rho - (circle[1] * np.cos(theta) + circle[0] * np.sin(theta))) < 0.5:  # 这里的0.5是一个容差值
                converging_lines += 1
        #这个阈值代表多少条线汇聚在一起才算是飞镖盘
        if converging_lines > 50:  # 可根据需要调整线条汇聚数量阈值
            dartboards.append(circle)

    return dartboards

def detect_dartboards_with_hough(frame, model, img_name, scaleFactor=1.008, minNeighbors=25, flags=0, minSize=(50, 50),
                                 maxSize=(300, 300), thresholdH=9,line_threshold=60):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Viola-Jones检测
    VioJonResult = model.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, flags=flags,
                                          minSize=minSize, maxSize=maxSize).tolist()
    VioJonResult = non_maximum_suppressionForViola(VioJonResult, 0.2)

    print(f"Detected {len(VioJonResult)} VioJonResult")

    # for i in range(0, len(VioJonResult)):
    #     start_point = (VioJonResult[i][0], VioJonResult[i][1])
    #     # VioJonResult[i][0] + VioJonResult[i][2] = x+ width
    #     end_point = (VioJonResult[i][0] + VioJonResult[i][2], VioJonResult[i][1] + VioJonResult[i][3])
    #     colour = (0, 255, 0)
    #     # the thickness of the line
    #     thickness = 2
    #     frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

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

    # 霍夫变换
    AfterBlur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
    DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
    Direction = np.arctan2(DerivativeY, DerivativeX)
    cv2.imwrite(f"task3copy/threshold/threshold_{img_name}.jpg",
                threshold_image(cv2.imread(f"task3copy/magnitude/Magnitude_{img_name}.jpg", cv2.IMREAD_GRAYSCALE),
                                40))

    thresholdImgWhole = cv2.imread(f"task3copy/threshold/threshold_{img_name}.jpg", cv2.IMREAD_GRAYSCALE)
    rectangle_for_circle = []
    total_dartboard_circles = []
    # 设定扩展大小,我希望在比每个viola-johns框稍微大一点的区域内做霍夫变换
    expand_size = 25  # 您可以根据需要调整这个值
    # 对于每个检测到的飞镖盘，执行霍夫变换来找圆
    for (x, y, w, h) in VioJonResult:
        # 计算新的裁剪坐标，确保坐标不超出图像边界
        x_start = max(x - expand_size, 0)
        y_start = max(y - expand_size, 0)
        x_end = min(x + w + expand_size, frame_gray.shape[1])
        y_end = min(y + h + expand_size, frame_gray.shape[0])
        thresholdImg = thresholdImgWhole[y_start:y_end, x_start:x_end]
        Direction_cropped = Direction[y_start:y_end, x_start:x_end]

        # 应用霍夫变换
        hough_space = houghTransformation(thresholdImg, Direction_cropped, 20, 200)

        # 从霍夫空间中找到可能的圆
        dartboard_circles = detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold)
        for board in dartboard_circles:
            # 调整圆的坐标，以对应到原图的全局坐标
            total_dartboard_circles.append((board[0] + y_start, board[1] + x_start, board[2]))

        # 裁剪检测区域
        # thresholdImg = thresholdImgWhole[y:y + h, x:x + w]
        # # 应用霍夫变换
        # hough_space = houghTransformation(thresholdImg, Direction[y:y + h, x:x + w], 10, 200)
        # # 从霍夫空间中找到可能的圆
        # dartboard_circles = find_parameter(hough_space, thresholdH)
        # for board in dartboard_circles:
        #     # 这个地方要变成全局坐标
        #     total_dartboard_circles.append((board[0] + y, board[1] + x, board[2]))

    #我发现不加这一句效果更好，原因其实就是我直接过滤大的圆，可能会导致真正在框中很多的圆被过滤掉
    # total_dartboard_circles = filter_similar_circles(total_dartboard_circles)
    result_circle = []
    result_rectangle_for_circle = []
    for board in total_dartboard_circles:
        # 把每个圆做成一个precision方框，以用来计算
        top_left_x = board[1] - board[2]
        top_left_y = board[0] - board[2]
        width_height = 2 * board[2]
        rectangle_for_circle.append((top_left_x, top_left_y, width_height, width_height))
    # 我还可以对每个viola-jones区域都只保留一个最大的圆,计算圆和viola方框的IOU
    for Board in VioJonResult:
        best_iou = 0
        best_pred_index = -1
        for i, pred in enumerate(rectangle_for_circle):
            iou = computeIoU(Board, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_index = i
        if best_iou > 0.3 :
            # 可以理解为这两个下标是共享的
            result_rectangle_for_circle.append(rectangle_for_circle[best_pred_index])
            result_circle.append(total_dartboard_circles[best_pred_index])

#我在这里再过滤相似的圆
    for board in result_circle:
        cv2.circle(frame, (board[1], board[0]), board[2], (0, 255, 255), 2)

    TPR, F1 = evaluate_predictions(result_rectangle_for_circle, groundtruth[current_image_name])
    print("TPR: ", TPR)
    print("F1: ", F1)
    with open(f"task3copy/evaluation_results.txt", 'a') as file:
        file.write(f"TPR: {TPR}\n")
        file.write(f"F1: {F1}\n")
    # 返回带有标记圆的图像
    return frame


def non_maximum_suppressionForViola(boxes, iou_threshold):
    # 按面积对边界框进行降序排序
    boxes = sorted(boxes, key=lambda x: x[2] * x[3], reverse=True)

    # 初始化最终的边界框列表
    final_boxes = []

    # 遍历每个边界框
    for current_box in boxes:
        # 对于列表中的第一个边界框（即面积最大的），直接添加到最终列表中
        if not final_boxes:
            final_boxes.append(current_box)
            continue

        # 计算当前边界框与最终列表中每个边界框的IoU，并确保IoU都小于阈值
        keep_current_box = True
        for final_box in final_boxes:
            if is_inside(current_box, final_box):
                keep_current_box = False
                break

            if computeIoU(current_box, final_box) > iou_threshold:
                keep_current_box = False
                break

        # 如果当前边界框的IoU都小于阈值，则将其添加到最终列表中
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

    # TP 是人脸，预测也是人脸
    TP = 0
    # FP 不是人脸，预测是人脸 假阳
    FP = 0
    # FN 是人脸，预测不是人脸 假阴
    FN = 0

    # For each ground truth, find the best matching prediction
    # 对每个ground truth，找到最匹配的prediction,这样就可以防止一个ground truth里面有多个prediction匹配到它，造成TP非常多
    for gt in ground_truths:
        best_iou = 0
        best_pred_index = -1
        isInside = False
        # 遍历prediction，找到和当前ground truth最匹配的prediction，也就是IOU最大的prediction
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
        # 找到了这个ground truth对应的最好的prediction以后，然后再判断他们的IOU是否大于阈值，如果大于阈值，就是TP，否则就是FN
        if best_iou > iou_threshold or isInside:
            TP += 1
            # Remove this prediction from further consideration
            # 匹配成功了，就把这个prediction从predictions里面移除，因为一个prediction只能匹配一个ground truth
            predictions_copy.pop(best_pred_index)
        else:
            FN += 1

    # Any remaining predictions are FP
    # 这些都不是人脸
    FP = len(predictions_copy)

    # Calculate TPR (true positive rate)
    TPR = TP / (TP + FN)

    # Calculate precision and recall
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TPR

    # Calculate F1 score
    F1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0

    return TPR, F1


# ************ NEED MODIFICATION ************
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

if os.path.exists("task3copy/evaluation_results.txt"):
    # If it does, delete the file
    os.remove("task3copy/evaluation_results.txt")

image_files = [f"Dartboard/dart{i}.jpg" for i in range(16)]  # 生成包含dart0.jpg到dart4.jpg的列表

# image_files = [f"Dartboard/dart3.jpg" ]  # 生成包含dart0.jpg到dart4.jpg的列表
# image_files = ["Dartboard/dart13.jpg","Dartboard/dart3.jpg","Dartboard/dart6.jpg","Dartboard/dart9.jpg","Dartboard/dart11.jpg"]

for image_file in image_files:
    # 提取文件名（不含扩展名）作为后续保存文件的一部分
    image_name = os.path.splitext(os.path.basename(image_file))[0]

    # 检查文件是否存在
    if not os.path.isfile(image_file):
        print(f'No such file: {image_file}')
        continue

    # 读取图片
    frame = cv2.imread(image_file, 1)
    if not (type(frame) is np.ndarray):
        print(f'Not image data: {image_file}')
        continue

    # 加载Viola-Jones模型
    cascade_name = "Dartboardcascade/cascade.xml"
    model = cv2.CascadeClassifier()
    if not model.load(cascade_name):
        print('--(!)Error loading cascade model')
        continue

    # 执行边缘检测并保存结果
    sobelEdgeDetect(frame, image_name)  # 如果这个函数内部保存了结果，确保它使用image_name来命名输出文件

    # 执行Viola-Jones和霍夫变换
    result_image = detect_dartboards_with_hough(frame, model, image_name)

    # 保存检测结果图像时，包含图片的编号
    cv2.imwrite(f"task3copy/detected_{image_name}.jpg", result_image)
