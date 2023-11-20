import cv2
import numpy as np

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


def draw_lines(image, accumulator, thetas, rhos, threshold):
    # np.where函数找出累加器中所有值大于给定阈值threshold的元素的索引。
    # idxs是一个元组，包含了两个数组，第一个数组是行索引，第二个数组是列索引。
    # idxs[0][]包含了所有的rho，idxs[1][]包含了所有的theta。
    idxs = np.where(accumulator > threshold)
    for i in range(len(idxs[0])):
        rho = rhos[idxs[0][i]]
        theta = thetas[idxs[1][i]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
    return image

# Example usage:

img = cv2.imread("lineExample.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imwrite("lineedges.jpg", edges)
accumulator, thetas, rhos = hough_line_transform(edges)
result = draw_lines(img, accumulator, thetas, rhos, 50)
cv2.imwrite("lineresult.jpg", result)





