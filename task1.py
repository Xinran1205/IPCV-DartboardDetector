################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse
import copy

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='dart detection')
# Updated to accept multiple filenames
parser.add_argument('-names', nargs='+', default=['Dartboard/dart0.jpg','Dartboard/dart1.jpg','Dartboard/dart2.jpg',
                                                  'Dartboard/dart3.jpg','Dartboard/dart4.jpg','Dartboard/dart5.jpg',
                                                'Dartboard/dart6.jpg','Dartboard/dart7.jpg','Dartboard/dart8.jpg',
                                                'Dartboard/dart9.jpg','Dartboard/dart10.jpg','Dartboard/dart11.jpg','Dartboard/dart12.jpg',
                                                'Dartboard/dart13.jpg','Dartboard/dart14.jpg','Dartboard/dart15.jpg'])
args = parser.parse_args()

# Global variables
cascade_name = "Dartboardcascade/cascade.xml"
model = cv2.CascadeClassifier(cascade_name)


# /** Global variables */
cascade_name = "Dartboardcascade/cascade.xml"


def detectAndDisplay(frame):

	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection

    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.008, minNeighbors=25, flags=0, minSize=(50,50), maxSize=(300,300)).tolist()

    # 3. Print number of dartboards found
    print(len(dartboards))

    for i in range(0, len(dartboards)):
        start_point = (dartboards[i][0], dartboards[i][1])
        # dartboards[i][0] + dartboards[i][2] = x+ width
        end_point = (dartboards[i][0] + dartboards[i][2], dartboards[i][1] + dartboards[i][3])
        colour = (0, 255, 0)
        # the thickness of the line
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    current_image_name = os.path.splitext(imageName.split('/')[-1])[0]

    #Draw groundtruth
    groundtruth = readGroundtruth()
    for img_name in groundtruth:
        if img_name == current_image_name:
            for bbox in groundtruth[img_name]:
                start_point = (bbox[0], bbox[1])
                end_point = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                colour = (0, 0, 255)
                thickness = 2
                frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    TPR, F1 = evaluate_predictions(dartboards, groundtruth[current_image_name])
    print("TPR: ", TPR)
    print("F1: ", F1)


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


def evaluate_predictions(predictions, ground_truths, iou_threshold=0.2):
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
        # 遍历prediction，找到和当前ground truth最匹配的prediction，也就是IOU最大的prediction
        for i, pred in enumerate(predictions_copy):
            iou = computeIoU(gt, pred)
            if iou > best_iou:
                best_iou = iou
                best_pred_index = i

        # If the best matching prediction has IOU > threshold, it's a TP. Otherwise, it's a FN.
        # 找到了这个ground truth对应的最好的prediction以后，然后再判断他们的IOU是否大于阈值，如果大于阈值，就是TP，否则就是FN
        if best_iou > iou_threshold:
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


for imageName in args.names:
    # Check if the image and cascade files are present
    if not os.path.isfile(imageName) or not model.load(cascade_name):
        print('File not found:', imageName)
        continue  # Skip this image and continue with the next one

    # Read and process the image
    frame = cv2.imread(imageName, 1)
    if frame is None:
        print('Failed to load image:', imageName)
        continue

    # Detect dartboards and display results
    detectAndDisplay(frame)

    # Construct the new image name and save it in the task1output directory
    output_dir = 'task1output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    base_filename = os.path.basename(imageName)
    new_image_name = os.path.splitext(base_filename)[0] + '_detected.jpg'
    output_path = os.path.join(output_dir, new_image_name)

    cv2.imwrite(output_path, frame)
    print('Image saved:', output_path)



