"""
this file contains all of the constants. this is so we wouldn't have a loop in importing.
for example: main imports border and border imports main.
"""


# all of the classes that the net can detect
import sys

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

PROTOTXT__PATH = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"
CAFFE_MODEL_PATH = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
VID_PATH = "videos/"
OUTPUT_PATH = "output/"
VID_1_NAME = "3.mp4"
VID_2_NAME = "4.mp4"
CONFIDENCE = 0.4
SKIP_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ABSENCE_BEFORE_REMOVE = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MAX_DISTANCE_FROM_CENTROID = 50
UPPER_LINE_PLACE = 1.3
LOWER_LINE_PLACE = 7
