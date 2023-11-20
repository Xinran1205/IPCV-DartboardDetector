import cv2
import numpy as np
import os
import sys
import argparse
import copy

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


def double_thresholding(img, low_ratio=0.03, high_ratio=0.1):
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


def sobelEdgeDetect(img):
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
    Magnitude = np.sqrt(DerivativeX**2 + DerivativeY**2)
    normalized_magnitude = (Magnitude / np.max(Magnitude) * 255).astype(np.uint8)
    normalized_magnitude = non_maximum_suppression(normalized_magnitude, Direction)
    normalized_magnitude = double_thresholding(normalized_magnitude)
    cv2.imwrite("HoughMagnitude/Magnitude.jpg", normalized_magnitude)


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

def filter_largest_circles(parameters, center_error=10, radius_error=5):
    # 使用字典来按圆心分组圆
    circles_by_center = {}
    for y, x, r in parameters:
        # 查找已有圆心是否存在
        found = False
        for (cy, cx), (cr, _) in circles_by_center.items():
            if ((cy - y) ** 2 + (cx - x) ** 2) ** 0.5 <= center_error:
                found = True
                # 如果当前圆的半径比已记录的圆更大，则更新
                if r > cr and abs(r - cr) > radius_error:
                    circles_by_center[(cy, cx)] = (r, (y, x, r))
                break
        if not found:
            circles_by_center[(y, x)] = (r, (y, x, r))

    # 只保留每组中半径最大的圆
    return [info[1] for info in circles_by_center.values()]


def filter_similar_circles(parameters, center_threshold=20, radius_threshold=20):
    filtered_params = []

    for current_circle in parameters:
        is_similar = False
        for saved_circle in filtered_params:
            center_distance = ((current_circle[0] - saved_circle[0]) ** 2 +
                               (current_circle[1] - saved_circle[1]) ** 2) ** 0.5
            radius_diff = abs(current_circle[2] - saved_circle[2])

            if center_distance < center_threshold and radius_diff < radius_threshold:
                is_similar = True
                break
        # if the circle is not similar to any other circle, add it to the list
        if not is_similar:
            filtered_params.append(current_circle)

    return filtered_params


def draw_circle(image, parameters):
    parameters = filter_similar_circles(parameters)
    for param in parameters:
        cv2.circle(image, (param[1], param[0]), param[2], (0, 255, 255), 2)
    return image

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


def draw_lines(image, accumulator, thetas, rhos, threshold, vertex):
    idxs = np.where(accumulator > threshold)
    for i in range(len(idxs[0])):
        rho = rhos[idxs[0][i]]
        theta = thetas[idxs[1][i]]
        a = np.cos(theta)
        b = np.sin(theta)
        # 检查顶点是否在直线上
        # 直线的极坐标方程是 rho = x*cos(theta) + y*sin(theta)
        # 如果顶点满足这个方程，它就在直线上
        if abs(rho - (vertex[0]*a + vertex[1]*b)) < 15: # 这里的1是一个容差值
            # 顶点在直线上，绘制直线
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
    return image


def detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold):
    # 找到可能的圆
    potential_circles = find_parameter(hough_space, thresholdH)
    # 过滤相似的圆
    potential_circles = filter_similar_circles(potential_circles)

    potential_circles = filter_largest_circles(potential_circles)

    # 确定飞镖盘
    dartboards = []
    # 针对这块区域（viola检测出的区域）计算直线的霍夫变换
    accumulator, thetas, rhos = hough_line_transform(thresholdImg)
    for circle in potential_circles:
        idxs = np.where(accumulator > line_threshold)
        converging_lines = 0
        for i in range(len(idxs[0])):
            rho = rhos[idxs[0][i]]
            theta = thetas[idxs[1][i]]
            # 检查顶点是否在直线上
            # 直线的极坐标方程是 rho = x*cos(theta) + y*sin(theta)
            # 如果顶点满足这个方程，它就在直线上
            if abs(rho - (circle[1]*np.cos(theta) + circle[0]*np.sin(theta))) < 0.5:  # 这里的0.5是一个容差值
                converging_lines += 1
        if converging_lines > 30:  # 可根据需要调整线条汇聚数量阈值
            dartboards.append(circle)

    return dartboards

def detect_dartboards_with_hough(frame, model, scaleFactor=1.01, minNeighbors=20, flags=0, minSize=(50,50), maxSize=(500,500), thresholdH=10, line_threshold=80):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # Viola-Jones检测
    dartboards = model.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, flags=flags,
                                        minSize=minSize, maxSize=maxSize).tolist()

    print(f"Detected {len(dartboards)} dartboards")

    for i in range(0, len(dartboards)):
        start_point = (dartboards[i][0], dartboards[i][1])
        # dartboards[i][0] + dartboards[i][2] = x+ width
        end_point = (dartboards[i][0] + dartboards[i][2], dartboards[i][1] + dartboards[i][3])
        colour = (0, 255, 0)
        # the thickness of the line
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    # 霍夫变换
    AfterBlur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
    DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
    Direction = np.arctan2(DerivativeY, DerivativeX)
    cv2.imwrite("HoughThreshold/threshold.jpg", threshold_image(cv2.imread("HoughMagnitude/Magnitude.jpg", cv2.IMREAD_GRAYSCALE), 120))

    thresholdImgWhole = cv2.imread("HoughThreshold/threshold.jpg", cv2.IMREAD_GRAYSCALE)
    # 对于每个检测到的飞镖盘，执行霍夫变换来找圆
    for (x, y, w, h) in dartboards:
        # 裁剪检测区域
        thresholdImg = thresholdImgWhole[y:y + h, x:x + w]
        # 应用霍夫变换
        hough_space = houghTransformation(thresholdImg, Direction[y:y + h, x:x + w], 50, 300)
        # 从霍夫空间中找到可能的圆
        dartboard_circles = detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold)
        for board in dartboard_circles:
            # 画出每个飞镖盘的圆
            cv2.circle(frame, (board[1] + x, board[0] + y), board[2], (0, 255, 255), 2)

    # 返回带有标记圆的图像
    return frame

# 我在这个函数里面生成了magnitude图像
sobelEdgeDetect(cv2.imread("Dartboard/dart14.jpg"))

parser = argparse.ArgumentParser(description='Dartboard detection with Hough Transform')
parser.add_argument('-name', '-n', type=str, default='Dartboard/dart14.jpg')
args = parser.parse_args()

imageName = args.name
cascade_name = "Dartboardcascade/cascade.xml"

# 加载图像
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

frame = cv2.imread(imageName, 1)
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)

# 加载Viola-Jones模型
model = cv2.CascadeClassifier()
if not model.load(cascade_name):
    print('--(!)Error loading cascade model')
    sys.exit(1)

# 执行Viola-Jones和霍夫变换
result_image = detect_dartboards_with_hough(frame, model)

# 保存检测结果图像
cv2.imwrite("HoughResult/detected_dartboards.jpg", result_image)