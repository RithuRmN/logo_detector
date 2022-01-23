import pandas as pd
import argparse
import os
import numpy as np
 
parser = argparse.ArgumentParser()
parser.add_argument('--csv_root', default='experiment/OD_BOLT/TEMP/val.csv' )
parser.add_argument('--output_root', default='src/metric/mAP/results/GroundTruth')
args = parser.parse_args()

if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
#csv_root = '/home/dtai/Documents/Ritika/Ellie/Object_Detection/rfmb/
#output_root = '/home/dtai/Documents/Ritika/Ellie/Object_Detection/rfmb/

df_annotation = pd.read_csv(args.csv_root).drop(['Unnamed: 0'], axis = 1) #(FLAGS.annotation_csv)
gb = df_annotation.groupby('f_name')
grouped_list = [gb.get_group(x) for x in gb.groups]
for img_group in grouped_list:
    # print(img_group.columns)
    filepath = img_group['f_name'].iat[0].replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.png', '.txt').replace('.PNG', '.txt').replace('.jpeg', '.txt').replace('.JPEG', '.txt')
    pd_coordinates = img_group.drop(['f_name'],axis=1)
    # print(pd_coordinates)
    int_cols = ["xmin", "ymin","xmax", "ymax"]
    for i in int_cols:
        pd_coordinates[i] = pd_coordinates[i].astype(int)
      
    if pd_coordinates['xmin'].values[0] == 0 & pd_coordinates['xmax'].values[0]==0:
        pd_coordinates = pd.DataFrame()
        pd_coordinates.to_csv(os.path.join(args.output_root,filepath), header = False, index = False)
    else:    
        pd_coordinates = pd_coordinates.reset_index(drop=True)
        pd_coordinates = pd_coordinates[['label_type_name', 'xmin', 'ymin', 'xmax', 'ymax']]
        pd_coordinates.to_csv(os.path.join(args.output_root,filepath), header = False, index = False, sep = ' ')
    
