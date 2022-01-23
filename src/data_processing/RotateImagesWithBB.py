# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import imageio 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import matplotlib.pyplot as plt


# %%
import pandas as pd
import argparse
import os
import numpy as np
import argparse
import glob
# %%
# Set parser
parser = argparse.ArgumentParser()
parser.add_argument("--input_csv_path", default="./data/annotation/train.csv")
parser.add_argument("--output_csv_path", default="./data/annotation/train_rotated.csv")
parser.add_argument("--input_images_root", default='./data/Images')
parser.add_argument("--output_images_root", default="data/images_with_rotation")
args = parser.parse_args()



degree = 0
augmentations_list = [0,90,270]


# %%
if not os.path.exists(args.output_images_root):
    os.makedirs(args.output_images_root)


def find_match_folder_recursive(string_to_match):
    return [name for name in glob.glob(string_to_match, recursive= True)]
# %%
def get_info_for_one_group(df_annotation,group_f,images_path):
    # image_file = df_annotation['filename'].at[group_f[0]]
    bb_list = []
    #count = 0
    for indx in group_f:
        #print("working")
        image_file = df_annotation['filename'].at[indx]
        x1 = df_annotation['xmin'].at[indx] #group_f[count]
        x2 = df_annotation['xmax'].at[indx]
        y1 = df_annotation['ymin'].at[indx]
        y2 = df_annotation['ymax'].at[indx]
        #count+=1
        bb_list.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
    string_to_match = os.path.join(images_path,image_file.split('.')[0]+'*')
            
    filepath = find_match_folder_recursive(string_to_match)
    if filepath==[]:
        print("file not found")
        return None,None,None
    else:
        print(filepath[0])
        image_points = imageio.imread(os.path.join(filepath[0]),pilmode="RGB")
        bbs = BoundingBoxesOnImage(bb_list, shape=image_points.shape)
        #print(image_file)
        return image_file,bbs,image_points

def perform_rotation(degree,image_points,bbs):
    ia.seed(1)
    rot = iaa.Affine(rotate=degree,fit_output=True)
    image_points_aug, bbs_aug = rot(image=image_points, bounding_boxes=bbs)
    return image_points_aug, bbs_aug

def save_info_for_one_group(image_file,group_f,df_annotation_rotated,bbs_aug,image_points_aug,augmented_images_root,degree):
    count = 0
    #image_file = df_annotation['filename'].at[group_f[0]]
    new_image_name = image_file.split('.')[0]+"__"+str(degree)+"."+"jpg"
    for indx in group_f:
        bb_post_aug = bbs_aug.bounding_boxes[count]
        df_annotation_rotated['xmin'].at[indx] = bb_post_aug.x1
        df_annotation_rotated['xmax'].at[indx] = bb_post_aug.x2
        df_annotation_rotated['ymin'].at[indx] = bb_post_aug.y1
        df_annotation_rotated['ymax'].at[indx] = bb_post_aug.y2
        df_annotation_rotated['filename'].at[indx] = new_image_name
        count+=1
    
    imageio.imsave(os.path.join(augmented_images_root, new_image_name),image_points_aug)
    # print(new_image_name)
    return df_annotation_rotated 

def check_for_one_group(df_annotation_rotated,group_f,augmented_images_root):
    bbs_recreated = []
    for indx in group_f:
        image_file = df_annotation_rotated['filename'].at[indx]
        x1 = df_annotation_rotated['xmin'].at[indx]
        x2 = df_annotation_rotated['xmax'].at[indx]
        y1 = df_annotation_rotated['ymin'].at[indx]
        y2 = df_annotation_rotated['ymax'].at[indx]
        bbs_recreated.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2))
    bbs_recreated_ = BoundingBoxesOnImage(bbs_recreated, shape=image_points.shape)
    image_points_recreated = imageio.imread(os.path.join(augmented_images_root,image_file))
    #print("recreated",os.path.join(augmented_images_root,image_file))
    return bbs_recreated_, image_points_recreated

def perform_rotation_on_all(train_root,images_root,degree,augmented_images_root):
    df_annotation = pd.read_csv(train_root)
    df_annotation_rotated = df_annotation.copy()
    gb = df_annotation.groupby('filename')
    all_groups = [gb.groups[x] for x in gb.groups]
    for group_f in all_groups:
        #print(group_f['filename'].at[0])
        image_file,bbs,image_points = get_info_for_one_group(df_annotation,group_f,images_root)
        if image_file!=None:
            image_points_aug, bbs_aug = perform_rotation(degree,image_points,bbs)
            df_annotation_rotated = save_info_for_one_group(image_file,group_f,df_annotation_rotated,bbs_aug,image_points_aug,augmented_images_root,degree)

    return df_annotation_rotated


# %%
df_to_train_on = pd.read_csv(args.input_csv_path)
df_to_store_aug = pd.DataFrame(columns = df_to_train_on.columns)

for degree in augmentations_list:
    df_annotation_rotated = perform_rotation_on_all(args.input_csv_path,args.input_images_root,degree,args.output_images_root) 
    df_to_store_aug = pd.concat([df_to_store_aug,df_annotation_rotated]) 

df_to_store_aug.to_csv(args.output_csv_path)


# %%


