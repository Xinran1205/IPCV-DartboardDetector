import cv2
import numpy as np

# this class is just a draft to try to implement hough_line_transform before task3
# In task3 I write the new function hough_line_transform

def hough_line_transform(image):

    # Therefore, thetas is an array containing the radian values from -90 degrees to 89 degrees.
    # thetas[0] is the radian value of -90 degrees and -1.57
    thetas = np.deg2rad(np.arange(-90, 90))
    width, height = image.shape
    # calculate the length of the diagonal
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
    # x_idxs and y_idxs are of the same length, they are one-to-one pairs, each pair represents the coordinates of an edge point.
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


def draw_lines(image, accumulator, thetas, rhos, threshold):
    # np.where function finds the indexes of all elements in the accumulator that are greater than the given threshold.
    # idxs is a tuple containing two arrays, the first array is the row index, and the second array is the column index.
    # idxs[0][] contains all rho, idxs[1][] contains all theta.
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





