import cv2
import numpy as np
import glob
import random
import matplotlib.pyplot as plt

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["Head"]
out_dir = "yolo_results"
conf_threshold = 0.4
nms_threshold = 0.3
# Images path
#images_path = glob.glob("../input/object/interview_data/images/*.jpg")[1]
images_path = "test/4.jpg"
#print(images_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = (0,0,255)#np.random.uniform(0, 255, size=(len(classes), 3))
#print("in loop 0")

# Insert here the path of your images
#random.shuffle(images_path)
# loop through all the images
#for img_path in images_path:
# Loading image
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = colors#[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 3)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

img = cv2.imread(images_path)
#print("in loop 1")
#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
print(img.shape)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# initialization
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.4
nms_threshold = 0.3

# for each detetion from each output layer 
# get the confidence, class id, bounding box params
# and ignore weak detections (confidence < 0.5)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.42:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
# apply non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
for i in indices:
    i = i[0]
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    draw_bounding_box(img, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    # Save output Image
    cropped = img[round(y):round(y+h), round(x):round(x+w)]
    cropped=cv2.resize(cropped,(28,28))
    plt.imsave('{}/{}'.format(out_dir, images_path.split("/")[-1], cropped)
# display output image 
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


