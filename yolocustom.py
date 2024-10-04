import time
import os
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
from pycoral.utils.edgetpu import make_interpreter
from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
import tflite_runtime.interpreter as tflite
import cv2 as cv2
from pycoral.utils import edgetpu

from nms import non_max_suppression


def plot_one_box(box, im, color=(255, 0, 0), txt_color=(255, 255, 255), label=None, line_width=3, size=640):
    # Plots one xyxy box on image im with label
    # assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    lw = line_width or max(int(min(im.size) / 200), 2)  # line width

    c1, c2 = (int(box[0]*size), int(box[1]*size)), (int(box[2]*size), int(box[3]*size))

    cv2.rectangle(im, c1, c2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        txt_width, txt_height = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        c2 = c1[0] + txt_width, c1[1] - txt_height - 3
        #print("c1 is ", c1, " and c2 is ", c2)
        cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(im, label, (c1[0], c1[1] - 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
    return im

labels = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

set_verbosity(0)

delegates = [edgetpu.load_edgetpu_delegate(options=
                                           {
                                                "device":"usb:0", "Usb.MaxBulkInQueueLength": "256", "Performance": "Max", 
                                               })]

print(delegates[0])

interpreter =  tflite.Interpreter(model_path="yolov5s_oneanchor_10_classes_500_epoch_int8_edgetpu.tflite"
                                  , experimental_delegates=delegates)

# Load the TFLite model
#interpreter = make_interpreter("yolov5s_oneanchor_10_classes_500_epoch_edgetpu.tflite")

#interpreter = tf.lite.Interpreter(model_path="yolov5s_oneanchor_10_classes_500_epoch.tflite")
#exit()
interpreter.allocate_tensors()
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_zero = input_details[0]['quantization'][1]
input_scale = input_details[0]['quantization'][0]
output_zero = output_details[0]['quantization'][1]
output_scale = output_details[0]['quantization'][0]

# If the model isn't quantized then these should be zero
# Check against small epsilon to avoid comparing float/int
if input_scale < 1e-9:
    input_scale = 1.0

if output_scale < 1e-9:
    output_scale = 1.0

print("Input shape", input_details[0]['shape'])
print("Output shape", output_details[0]['shape'])

# Load and resize an image
img_path = "test.jpg"
img = Image.open(img_path).resize((640, 640))

img_array = np.expand_dims(img, axis=0)  # Add batch dimension

print(img_array.shape)


x = img_array.astype('float32')/255.0
#whax = img_array.astype('float32')

# Scale input, conversion is: real = (int_8 - zero)*scale
x = (x / input_scale) + input_zero
x = x.astype(np.int8)

conf_thres = 0.25
iou_thres = 0.45


for i in range(3):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()


#prediction = (prediction  * output_scale) + output_zero

    prediction = interpreter.get_tensor(output_details[0]['index']).astype('float32')
    prediction = (prediction - output_zero) * output_scale

    prediction = prediction.astype('float32')

    nms_result = non_max_suppression(prediction, conf_thres, iou_thres, None,
                                    False, max_det=300)
    print("Number of objects found:",nms_result[0].shape[0])

    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))

print(prediction[0].shape)


nms_result = non_max_suppression(prediction, conf_thres, iou_thres, None,
                                 False, max_det=1000)
print("Number of objects found:",nms_result[0].shape[0])

#print(nms_result[0])


img.save("output.jpg")
image = cv2.imread("output.jpg")



for i in range(nms_result[0].shape[0]):
    #print("image coordinates are: ", nms_result[0][i][:4])
    plot_one_box(nms_result[0][i][:4], image, label=labels[int(nms_result[0][i][5])] + " " + str(int(nms_result[0][i][4]*100)) + "%", size=640, line_width=1)
cv2.imwrite("drawn.jpg", image)