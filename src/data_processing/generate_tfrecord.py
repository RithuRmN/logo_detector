"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python src/generate_tfrecord.py --csv_input=data/structured/post_processed_labels/train_11_6.csv  --output_path=data/structured/tfrecords/12_6/train_strat_roi_12_6.tfrecord --image_dir data/PostProcessedImages
  python src/generate_tfrecord.py --csv_input=data/structured/post_processed_labels/test_11_6.csv  --output_path=data/structured/tfrecords/12_6/test_strat_roi_12_6.tfrecord --image_dir data/PostProcessedImages
  
  # Create test data:
  python generate_tfrecord.py --csv_input=/home/dtai/Documents/Ritika/Ellie/labels_vott/test-0_1_class.csv  --output_path=/home/dtai/Documents/Ritika/Ellie/Object_Detection/data/split-0/data/test-0_1_class.tfrecord --image_dir /home/dtai/Documents/Ritika/Ellie/labels_vott/0
python generate_tfrecord.py --csv_input=/home/dtai/Documents/Ritika/Ellie/labels_vott/test-0.csv  --output_path=/home/dtai/Documents/Ritika/Ellie/Object_Detection/split-0/data/test-0.tfrecord --image_dir /home/dtai/Documents/Ritika/Ellie/labels_vott/0
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import yaml
import os
import io
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import argparse
import yaml

from PIL import Image
#import Image
import sys
sys.path.append('')
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from gen_utils import find_match_folder_recursive, merge_dfs_left, subset_df_per_column, make_new_dir

# flags = tf.app.flags
# flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# flags.DEFINE_string('image_dir', '', 'Path to images')
# FLAGS = flags.FLAGS
parser = argparse.ArgumentParser()
parser.add_argument('--paramspath','-m', default= 'params.yaml')
args = parser.parse_args()

params = yaml.safe_load(open(args.paramspath))


def split(df, group):
    data = namedtuple('data', ['f_name', 'object'])
    gb = df.groupby(group)
    return [data(f_name, gb.get_group(x)) for f_name, x in zip(gb.groups.keys(), gb.groups)]

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'ROI':
        return 0
    else:
        return 2000

def create_tf_example(group, path):
    string_to_match = os.path.join(path,'*', '*', group.f_name.split('.')[0]+'*')
    filepath = find_match_folder_recursive(string_to_match)
    
    if filepath!=[] :
        im = Image.open(filepath[0])
        width, height = im.size
        if width < 200 & height <200 :
            print("------Image size too small-------")
            filepath=[]

    if filepath!=[] :
        
        print(filepath[0])
        with tf.gfile.GFile(filepath[0], 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.f_name.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []
        class_name='ROI'
        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            # classes_text.append(row['label_type_name'].encode('utf8'))
            # classes.append(class_text_to_int['label_type_name'])
            classes_text.append(class_name.encode('utf8'))
            classes.append(0)
            
        # print(classes)
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example
    else:
        print("image_not_found")
        return 0

def main(_):
    rootpath_for_splits=params['PrepareLabels']["rootpath_for_splits"]
    output_path=params['GenerateTFRecord']["output_path"]
    image_dir=params['GenerateTFRecord']["image_dir"]
    create_tfrecord = params['GenerateTFRecord']["create_tfrecord"]
    splits = params['GenerateTFRecord']['split']
   
    for split_name in splits:
        print(split_name)
        output_path=params['GenerateTFRecord']["output_path"]
        writer = tf.python_io.TFRecordWriter(os.path.join(output_path ,split_name+".tfrecord"))
        path = os.path.join(os.getcwd(),image_dir)
        examples = pd.read_csv(os.path.join(rootpath_for_splits,split_name+".csv"))
        grouped = split(examples, 'f_name')
        for group in grouped:
            tf_example = create_tf_example(group, path)
            if tf_example!= 0:
                writer.write(tf_example.SerializeToString())

        writer.close()
        output_path = os.path.join(os.getcwd(), output_path ,split_name+".tfrecord")
        
        print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    tf.app.run()