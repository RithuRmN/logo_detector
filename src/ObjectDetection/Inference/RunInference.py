# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from PIL import Image
# Set parser
parser = argparse.ArgumentParser()
parser.add_argument("--classname", default='src/ObjectDetection/Training/data/lines.names', help = "Path to class names")
parser.add_argument("--tfserving", default='src/ObjectDetection/Training/checkpoint/stage2/tf_serving', help = "Path to folder with model pb file")
parser.add_argument("--image", default='src/ObjectDetection/Training/data/test.jpg', help = "Path to image file")
args = parser.parse_args()


# %%
def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
    return img
    
def print_predictions(output):
    number_of_valid_detections = np.array(output['yolo_nms_3'])
    for i in range(number_of_valid_detections[0]):
        classscores = np.array(output['yolo_nms_1'])[0][i]
        normalized_bounding_boxes = np.array(output['yolo_nms'])[0][i]
        class_detected = np.array(output['yolo_nms_2'])[0][i]
        print(classscores, normalized_bounding_boxes, class_detected)


	# %%
loaded = tf.saved_model.load(args.tfserving)
infer = loaded.signatures["serving_default"]
print("model_loaded")

# %%
img_raw = tf.image.decode_image(
		open(args.image, 'rb').read(), channels=3)
img = tf.expand_dims(img_raw, 0)
img = transform_images(img, 416)
output = infer(img)
print_predictions(output)

boxes, scores, classes, nums = output['yolo_nms'],output['yolo_nms_1'],output['yolo_nms_2'],output['yolo_nms_3']
print(boxes[0],scores[0])
class_names = [c.strip() for c in open(args.classname).readlines()]
img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
#plt.imshow(img)
#plt.show()
img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
im = Image.fromarray(img)
im.save("your_file.png")
plt.imshow(img)
plt.show()
print("predicted")

	# %%


