from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import wx

import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def get_path():
    app = wx.App(None)
    style = wx.FD_OPEN | wx.DD_DIR_MUST_EXIST
    dialog = wx.DirDialog(None, 'Open', style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def get_file_path(wildcard):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def draw_and_crop_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop_img = img[x1y1[1]:x2y2[1], x1y1[0]:x2y2[0]]
        img = cv2.rectangle(img, x1y1, x2y2, (0, 255, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
    return img,crop_img
    
def crop_image(img,coord):
    x=coord[0]
    y=coord[1]
    w=(coord[2]-coord[0])
    h=(coord[3]-coord[1])
    crop_img = img[y:y+h, x:x+w]
    return crop_img

@app.get("/")
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", context= {"request": request}) 

@app.get("/input_click")
def select_input_image():
    image_selected = get_file_path('')
    print(image_selected)
    return {"path": image_selected} 

@app.get("/output_click")
def select_output_path():
    folder_selected = get_path()
    print(folder_selected)
    return {"path": folder_selected} 

@app.get("/open_folder")
def open_folder(request: Request):
    
    output_path=request.query_params['output_path']
    print(output_path)
    path = os.path.realpath(output_path)
    os.startfile(path)
    #os.system(f'start{os.path.realpath(path)}')
    return {"path": "Success"}
    
@app.get("/detect")
def detect(request: Request):
    image_path=request.query_params['input_path']
    output_path=request.query_params['output_path']
    print("-----------------Loading Model---------------------------")
    print("-----------------Loading Model---------------------------")
    print("-----------------Loading Model---------------------------")
    loaded = tf.saved_model.load("./model/tf_serving")
    infer = loaded.signatures["serving_default"]
    print("-----------------model_loaded-----------------------------")

    img_raw = tf.image.decode_image(
            open(image_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img_raw, 0)
    img = transform_images(img, 416)
    output = infer(img)
    boxes, scores, classes, nums = output['yolo_nms'],output['yolo_nms_1'],output['yolo_nms_2'],output['yolo_nms_3']
    print(boxes[0],scores[0],"NUMS : ",nums)
    if nums[0]>0:
        class_names = [c.strip() for c in open("./model/lines.names").readlines()]
        img = cv2.cvtColor(img_raw.numpy(),cv2.COLOR_RGB2BGR)
        img,crop_image = draw_and_crop_outputs(img, (boxes, scores, classes, nums), class_names)
        
        im = Image.fromarray(img)
        crop_image = Image.fromarray(crop_image)
        im.save(os.path.join(output_path,"output.png"))
        crop_image.save(os.path.join(output_path,"croped_logo.png"))

        print("predicted")
        return {"result": " Logo Detected and Saved"} 
    else:
        print("not predicted")
        return {"result": " Logo not detected"} 

