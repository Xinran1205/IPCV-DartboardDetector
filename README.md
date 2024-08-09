

# Dartboard Detection System

- zh_CN [简体中文](/README.zh_CN.md)
## Project Overview
This project develops a system using computer vision technology to detect dartboards, primarily utilizing the Viola-Jones detector and Hough Transform (including Hough Circle and Line Transforms). The innovation of the project lies in applying Hough Transform within the regions detected by the Viola-Jones detector, which not only enhances detection accuracy and efficiency but also ensures that the system's True Positive Rate (TPR) remains at 1, while increasing the average F1 score to 82%. All Hough Transformations and optimization code were personally written by me, without the use of any pre-existing libraries.

## Process Details
### 1. Preliminary Detection:
- Using a trained Viola-Jones detector to detect dartboards in 16 images, adjusting parameters in the `detectMultiScale` function (such as `minSize`, `maxSize`, `scaleFactor`, and `minNeighbors`) to ensure that the TPR of each image is close to 1, while trying to maximize the F1 score.
- Using IoU (Intersection over Union) to verify the match between detection boxes and actual targets (ground truth).

### 2. Optimization and Precision Detection:
- Hough Circle Transform is applied within the areas detected by Viola-Jones, significantly improving detection accuracy and efficiency.
- Techniques such as NMS (Non-Maximum Suppression), dual thresholding, and fine-tuning of detection areas are used to enhance detection precision and reduce false positives.
- Matching filtering ensures that only the most matching circles are retained in each detection area, and global filtering removes duplicate or similar circles.

### 3. Integrating Hough Line Transform:
- A new Hough Line Transform is added to analyze each detected circle's area to confirm whether a sufficient number of lines converge at the circle center.
- If the number of lines does not meet the predefined threshold, that circle is discarded, further enhancing the system's selection precision.

## Technical Explanation and Optimization Measures
### Reasons for Using Hough Transform in Viola-Jones Detected Areas
1. **High TPR to Avoid Missed Detections**: I initially cover all potential target areas using the Viola-Jones detector, ensuring that each actual target overlaps at least one detection box, thus applying the Hough Circle Transform within these boxes effectively prevents missed detections.
2. **Improved Algorithm Efficiency**: I perform the Hough Circle Transform only within the areas detected by Viola-Jones, not the entire image, which significantly speeds up the process. This localized processing not only saves computational resources but also greatly reduces running time.
3. **Localized Processing for Precision**: I apply Hough Transform within localized areas, allowing for more precise processing of these regions identified as potentially containing targets. Using the IoU to assess the accuracy of each circular detection result, I can accurately calculate the TPR and F1 scores.

### Further Optimization Methods
1. **Application of Non-Maximum Suppression (NMS)**: I apply NMS to the areas recognized by the Viola-Jones algorithm, effectively reducing repeated detection of the same area in the image and avoiding redundant detection boxes.
2. **Using Sobel Operator and NMS**: I utilize the Sobel operator to calculate edge gradients and apply a specific NMS method to ensure that only the most significant edge points are retained, reducing thick edge issues and enhancing the clarity of edge detection.
3. **Dual Threshold Method**: Before performing the Hough Transform, I apply a dual threshold method to the gradient magnitudes to distinguish between strong and weak edges, retaining only the prominent strong edges, thereby enhancing the accuracy of the Hough Transform.
4. **Matching Filtering and Global Filtering**: In each Viola-Jones detection box, I retain only the most matching circle. Finally, I filter all circles globally again, removing smaller or similar circles based on similarity (IoU calculation), prioritizing larger circles, effectively preventing the situation where a large circle encompasses a smaller one.

### Detector Detection Results Example
<img src="/pic/1.png" alt="sample" width="250" height="180">
<img src="/pic/2.png" alt="sample" width="250" height="180">