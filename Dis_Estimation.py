import cv2 as cv
import numpy as np

# Distance constants (These are in cm for consistency with camera calibration)
PERSON_WIDTH = 40  # CM
MOBILE_WIDTH = 7.5  # CM
BOTTLE_WIDTH = 7  # CM (Example value, adjust based on actual bottle width)

# Object detector constants
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Ensure that 'bottle' is included in the classes.txt and the YOLO model is trained for it

# Setting up YOLO model
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load calibration data
calibration_data = np.load('MultiMatrix_2.npz')
camera_matrix = calibration_data['camMatrix']
dist_coeffs = calibration_data['distCoef']

# Focal length from camera matrix
focal_length_x = camera_matrix[0, 0]
focal_length_y = camera_matrix[1, 1]
focal_length = (focal_length_x + focal_length_y) / 2  # Averaging x and y focal lengths

# Object detector function
def object_detector(image):
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)

        # Draw rectangle and label
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-14), FONTS, 0.5, color, 2)

        # Collecting data for distance calculation
        if classid == 0:  # Person class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 67:  # Cell phone class id
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
        elif classid == 39:  # Bottle class id (Update with the correct id for 'bottle')
            data_list.append([class_names[classid], box[2], (box[0], box[1]-2)])
    return data_list

# Distance finder function using focal length from camera calibration
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

# Start video capture
cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_length, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_length, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'bottle':
            distance = distance_finder(focal_length, BOTTLE_WIDTH, d[1])
            x, y = d[2]
        
        cv.rectangle(frame, (x, y-3), (x+150, y+23), BLACK, -1)
        cv.putText(frame, f'Distance: {round(distance, 2)} cm', (x+5, y+13), FONTS, 0.48, GREEN, 2)

    cv.imshow('frame', frame)
    
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()



