# CUSTOM VERSION


import time
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycoral.utils.edgetpu import make_interpreter
from pycoral.pybind._pywrap_coral import SetVerbosity as set_verbosity
import tflite_runtime.interpreter as tflite
import cv2 as cv2
from pycoral.utils import edgetpu
import collections

import numpy as np
from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

from nms import non_max_suppression_v8

# -----------------------------------------------------------------------------------------------------------
def tiles_location_gen(img_size, tile_size, overlap):
  """Generates location of tiles after splitting the given image according the tile_size and overlap.

  BTW tiles = slices (as in for slicing)
  Will later be used in 
  
  Args:
    img_size (int, int): size of original image as width x height.
    tile_size (int, int): size of the returned tiles as width x height.
    overlap (int): The number of pixels to overlap the tiles.

  Yields:
    A list of points representing the coordinates of the tile in xmin, ymin,
    xmax, ymax.
  """

  tile_width, tile_height = tile_size
  img_width, img_height = img_size
  h_stride = tile_height - overlap
  w_stride = tile_width - overlap
  for h in range(0, img_height - 1, h_stride):
    for w in range(0, img_width - 1, w_stride):
      xmin = w
      ymin = h
      xmax = min(img_width, w + tile_width)
      ymax = min(img_height, h + tile_height)
      yield [xmin, ymin, xmax, ymax]


tile_sizes = "256x256"  # TODO: Change this cuz it's quite big for a slice
tile_overlap = 128
score_threshold = 0.5   # Asta e diferit de conf_treshold? Nu e folosit nicaieri  TODO: Remove it
iou_threshold = 0.1


def plot_one_box(box, im, color=(255, 0, 0), txt_color=(255, 255, 255), label=None, line_width=3, size=640):
    # E acelasi plot_one_box din yolocustomv8

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

delegates = [edgetpu.load_edgetpu_delegate(options=
                                           {
                                               # "device":"pci", "Usb.MaxBulkInQueueLength": "256", "Performance": "Max", 
                                               # sau "device":"usb:0", "Usb.MaxBulkInQueueLength": "256", "Performance": "Max",
                                               })]


interpreter =  tflite.Interpreter(model_path="model24sept.tflite"
                                  , experimental_delegates=delegates)
interpreter.allocate_tensors()
# interpreter.allocate_tensors()  # is the second time needed? TODO: Remove it.

labels = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]

img = Image.open("highway.jpg").convert('RGB')
draw = ImageDraw.Draw(img)  # TODO: Remove this, it is not used anywhere

countTiles = 1

img_size = img.size
tile_sizes = [map(int, tile_size.split('x')) for tile_size in tile_sizes.split(',')] # tile_size looks like: 256x256


# since in our case we just have a 256x256 tile size, the first for (outer) will have one iteration
# iterating trough tiles
for i, tile_location in enumerate(tiles_location_gen(img_size, (256, 256), tile_overlap)):  # moving to each tile's location 
    tile = img.crop(tile_location)   # cropping the image so that only the current tile is left

    # We need to resize the tile in order to make sure it dimension mismatch does not occur
    #tile = tile.resize((256, 256), Image.NEAREST) 
  
    """
    _, scale = common.set_resized_input(
        interpreter, tile.size,
        lambda size, img=tile: img.resize(size, Image.NEAREST))  # this will resize the image based on the needed scale: similar to Image.open(img_path).resize((256, 256))
    """
    path = "drawn_" + str(i) + ".jpg"

    tile = np.array(tile)  # getting tile in np array format
    #x = np.expand_dims(tile, axis=0) # expanding axis to match tile

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(tile_location)

    # moved before invoke() so the model gets metadata about tensores then inferes
    input_details = interpreter.get_input_details()  
    input_zero = input_details[0]['quantization'][1]
    input_scale = input_details[0]['quantization'][0]

    interpreter.invoke()   # triggering the inference process, TODO: is it needed here?

    #get output details, may be better before invoke to prepare the shape
    output_details = interpreter.get_output_details()   
    output_zero = output_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
  
    # If the model isn't quantized then these should be zero
    # Check against small epsilon to avoid comparing float/int
    if input_scale < 1e-9:
        input_scale = 1.0
  
    if output_scale < 1e-9:
        output_scale = 1.0

    tile = np.array(tile)  # getting tile in np array format
    x = np.expand_dims(tile, axis=0) # expanding axis to match tile
    x = x.astype('float32')/255.0  # converting to float
    #x = img_array.astype('float32')

    # Scale input, conversion is: real = (int_8 - zero)*scale
    x = (x / input_scale) + input_zero
    x = x.astype(np.int8)  # converting to int8

    # Prepare confidence treshold and IoU
    conf_thres = 0.01    # !!!! confidence threshold !!!!
    iou_thres = 0.45
  
    # acum facem inference si aplicam NMS ca si la yolocustom8.py


    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

  
    #prediction = (prediction  * output_scale) + output_zero
  
    prediction = interpreter.get_tensor(output_details[0]['index']).astype('float32')
    prediction = (prediction - output_zero) * output_scale

    inference_time = time.perf_counter() - start
    print('%.1fms' % (inference_time * 1000))

    """
    # Nu e neaparat sa trecem de 5 ori prin inference

    for i in range(1,5):
  
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
  
  
        #prediction = (prediction  * output_scale) + output_zero
  
        prediction = interpreter.get_tensor(output_details[0]['index']).astype('float32')
        prediction = (prediction - output_zero) * output_scale
  
        inference_time = time.perf_counter() - start
        print('%.1fms' % (inference_time * 1000))
    """
  
  
    prediction = interpreter.get_tensor(output_details[0]['index']).astype('float32')
    print("Shape of prediction", prediction[0].shape)
    prediction = prediction.transpose(0, 2, 1)
    print("Shape of prediction after transpose", prediction[0].shape)
    
    print("output scale", output_scale)
    print("output zero", output_zero)
  
    # Scale back, conversion is: real = (int_8 - zero)*scale
  
    prediction = (prediction - output_zero) * output_scale
    print(prediction[0][:][1])
  
    nms_result = non_max_suppression_v8(prediction, conf_thres, iou_thres, None,
                                  False, max_det=300)
  
    #print(f'HERE WE ARE WITH ITERATION {i}')
    print(f"Number of objects found for tile {i}: {nms_result[0].shape[0]}")
    for i in range(nms_result[0].shape[0]):
      print("image coordinates are: ", nms_result[0][i][:4])
      plot_one_box(nms_result[0][i][:4], tile, label=labels[int(nms_result[0][i][5])] + " " + str(int(nms_result[0][i][4]*100)) + "%", size=640, line_width=1)
    
    cv2.imwrite(path, tile)

    print(tile_location)
    
    
    
    """
    # there will be different drawn jpg's because we are working with tiles, not only the whole image as we did in yolocustom8.py
    path = "drawn_" + str(i) + ".jpg"
    cv2.imwrite(path, tile)


    """