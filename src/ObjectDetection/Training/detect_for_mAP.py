'''
python detect_for_mAP.py \
	--paramspath experiment\OD_2LINES\params.yaml

'''


import time
import pandas as pd
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
import yaml
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs, get_outputs_for_mAP

import mlflow


flags.DEFINE_string('paramspath', '', 'path to params')


def write_prediction_txt(output_file_pathh,w,h,boxes, scores, classes,nums,class_names,file,df):
    #if nums[0]!=0:
    with open(output_file_pathh, 'w') as out_f:
        #print(f"Numns --------------{nums}")
        i=0
        for i in range(nums[0]):
            #print(f"x_min is ---------------------------{np.array(boxes[0][0])*w}")
            x1,y1,x2,y2 = np.array(boxes[0][i])
            x_min = x1*w
            y_min = y1*h
            x_max = x2*w
            y_max = y2*h
            a = np.round(np.array(scores[0][i]),6)
            #print(f"x_min is ---------------------------{x_min}")
            out_box = '{} {} {} {} {} {}'.format(class_names[int(classes[0][i])].strip(), str(a),  int(x_min), int(y_min), int(x_max), int(y_max))
            # print(np.array(scores[0][i]))
            # print("$################$")
            # print(a)
            # print(out_box)
            out_f.write(out_box + "\n")
            df=df.append([[file,(class_names[int(classes[0][i])].strip()),int(x_min),int(y_min),int(x_max),int(y_max)]])
        if nums ==0 :
            x_min, y_min, x_max, y_max,class_name='No','No','No','No','No' 
            df=df.append([[file,class_name,x_min,y_min,x_max,y_max]]) 
    return df

def find_match_folder_recursive(string_to_match):
    return [name for name in glob.glob(string_to_match, recursive= True)]

def main(_argv):
    params = yaml.safe_load(open(FLAGS.paramspath))

    img_dir_path = params['Inference']['img_dir_path']
    tiny = params['Inference']['tiny']
    weights = params['Inference']['weights']
    classes = params['Inference']['classes']
    size = params['Inference']['size']
    num_classes = params['Inference']['num_classes']
    output_path = params['Inference']['output_path']
    annotation_csv = params['Inference']['annotation_csv']
    tfrecord = params['Inference']['tfrecord']

    output_txt=os.path.join(output_path,"txt")
    output_img=os.path.join(output_path,"img")      
    df = pd.DataFrame()
    # os.chdir(os.path.join(os.getcwd(),'models/yolov3'))
    if not os.path.exists(output_txt):
        os.makedirs(output_txt)

    if not os.path.exists(output_img):
        os.makedirs(output_img)

    df_annotation = pd.read_csv(annotation_csv)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if tiny:
        yolo = YoloV3Tiny(classes=num_classes)
    else:
        yolo = YoloV3(classes=num_classes)

    yolo.load_weights(weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(classes).readlines()]
    logging.info('classes loaded')

    if tfrecord:
        dataset = load_tfrecord_dataset(
            tfrecord, classes, size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(10)))
        # print("length of dataset",len(dataset))
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, size)
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))
        logging.info('detections:')
        # print(boxes)
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                        np.array(scores[0][i]),
                                        np.array(boxes[0][i])))
        
        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(os.path.join(output_img,"output.jpg"), img)
        logging.info('output saved to: {}'.format(output_img))

        # img_cv2 = cv2.imread(path_for_single_image) 
        # h = img_cv2.shape[0]
        # w = img_cv2.shape[1]
        # output_file_pathh = os.path.join(output_txt,file).replace('.jpg', '.txt').replace('.JPG', '.txt')
        # write_prediction_txt(output_file_pathh,w,h,boxes, scores, classes,nums,class_names) 
    else:
        #for file in os.listdir(FLAGS.image)[0:10]:
        for file in df_annotation.f_name: 
            
            string_to_match = os.path.join(img_dir_path,'*',file.split('.')[0]+'*')
            
            filepath = find_match_folder_recursive(string_to_match)
            if filepath==[]:
                print("file not found")
                print(string_to_match)
            else:
                path_for_single_image = filepath[0]
                img_raw = tf.image.decode_image(open(path_for_single_image, 'rb').read(), channels=3)
                img = tf.expand_dims(img_raw, 0)
                img = transform_images(img, size)
                t1 = time.time()
                boxes, scores, classes, nums = yolo(img)
                t2 = time.time()
                logging.info('time: {}'.format(t2 - t1))
                # print('detections:',boxes)
                for i in range(nums[0]):
                    logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))
                
                img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(os.path.join(output_img,file), img)
                logging.info('output saved to: {}'.format(output_img))

                img_cv2 = cv2.imread(path_for_single_image) 
                h = img_cv2.shape[0]
                w = img_cv2.shape[1]
                output_file_pathh = os.path.join(output_txt,file).replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.png', '.txt').replace('.PNG', '.txt').replace('.jpeg', '.txt').replace('.JPEG', '.txt')
                df=write_prediction_txt(output_file_pathh,w,h,boxes, scores, classes,nums,class_names,file,df)            
                
                #x_min, y_min, x_max, y_max = 
                ##print(type(x_min))
                #x1y1 = int(x_min),int(y_min)#.astype(np.int32) #TODO check if coordinates are correct
                #x2y2 = (int(x_max),int(y_max))#.astype(np.int32)
                #img = cv2.rectangle(img, (1,4),(30,40),(255, 0, 0), 2)#x1y1, x2y2, (255, 0, 0), 2)
                #img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
                #print(get_outputs_for_mAP(img, (boxes, scores, classes, nums), class_names))
                #cv2.imwrite(os.path.join(FLAGS.output,file), img)
                #logging.info('output saved to: {}'.format(FLAGS.output))
    print(df.head())
    df=df.drop_duplicates()
    df.to_csv(os.path.join(output_path,"prediction.csv"),header = ['f_name','label_type_name', 'xmin', 'ymin','xmax', 'ymax'] )
    
    # with mlflow.start_run(run_name="od_bolt"):
    #     mlflow.log_metric(key="accuracy",value=accuracy_var)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


