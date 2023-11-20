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


def double_thresholding(img, low_ratio=0.03, high_ratio=0.2):
    high_threshold = img.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    M, N = img.shape
    result = np.zeros((M, N), dtype=np.uint8)

    strong, weak = 255, 50

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

def hough_line_accumulator(img, theta_res=1, rho_res=1):
    height, width = img.shape
    max_rho = int(np.sqrt(height**2 + width**2))
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))
    rhos = np.arange(-max_rho, max_rho, rho_res)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # find all edge (nonzero) pixel indexes

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)):
            rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + max_rho)
            accumulator[rho, j] += 1

    return accumulator, thetas, rhos

def hough_line_peaks(accumulator, thetas, rhos, n_peaks, threshold=None):
    peaks = []
    accumulator = np.copy(accumulator)
    if threshold is None:
        threshold = 0.5 * accumulator.max()

    for _ in range(n_peaks):
        rho_idx, theta_idx = np.unravel_index(accumulator.argmax(), accumulator.shape)
        if accumulator[rho_idx, theta_idx] > threshold:
            peaks.append((rhos[rho_idx], thetas[theta_idx]))
            accumulator[rho_idx, theta_idx] = 0
        else:
            break
    return peaks


def detect_dartboard(hough_space, thresholdImg, thresholdH, line_threshold):
    # 找到可能的圆
    potential_circles = find_parameter(hough_space, thresholdH)
    # 过滤相似的圆
    potential_circles = filter_similar_circles(potential_circles)

    potential_circles = filter_largest_circles(potential_circles)

    # 确定飞镖盘
    dartboards = []
    for circle in potential_circles:
        # 针对每个圆计算直线的霍夫变换
        accumulator, thetas, rhos = hough_line_accumulator(thresholdImg)
        peaks = hough_line_peaks(accumulator, thetas, rhos, n_peaks=30, threshold=line_threshold)

        # 计算汇聚于圆心的直线数量
        converging_lines = 0
        for rho, theta in peaks:
            # 计算直线与圆心的最短距离
            distance = abs(rho - (circle[1]*np.cos(theta) + circle[0]*np.sin(theta)))
            if distance < 5:  # 可根据需要调整距离阈值
                converging_lines += 1

        # 如果汇聚的直线足够多，认为是飞镖盘
        if converging_lines > 5:  # 可根据需要调整线条汇聚数量阈值
            dartboards.append(circle)

    return potential_circles

sobelEdgeDetect(cv2.imread("Dartboard/dart14.jpg"))
img = cv2.imread("Dartboard/dart14.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
AfterBlur = cv2.GaussianBlur(img, (7, 7), 0)
DerivativeX = cv2.Sobel(AfterBlur, cv2.CV_64F, 1, 0, ksize=3)
DerivativeY = cv2.Sobel(AfterBlur, cv2.CV_64F, 0, 1, ksize=3)
Direction = np.arctan2(DerivativeY, DerivativeX)

cv2.imwrite("HoughThreshold/threshold.jpg", threshold_image(cv2.imread("HoughMagnitude/Magnitude.jpg", cv2.IMREAD_GRAYSCALE), 150))
# 先进行霍夫变换，找到可能的圆
hough_space = houghTransformation(cv2.imread("HoughThreshold/threshold.jpg", cv2.IMREAD_GRAYSCALE), Direction, 80, 130)
# # 然后在可能的圆中找直线，确定飞镖盘
dartboards = detect_dartboard(hough_space,cv2.imread("HoughThreshold/threshold.jpg", cv2.IMREAD_GRAYSCALE), 12, 30)
result_image = cv2.imread("Dartboard/dart14.jpg")
for board in dartboards:
    result_image = draw_circle(result_image, [board])

# 保存检测飞镖盘的结果图像
cv2.imwrite("onlyHoughResult/dartboard_detected14.jpg", result_image)
