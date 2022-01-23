import glob 
import cv2 
import pandas as pd
import os


def get_filenames_from_folder(PATH):
  ''' eg: PATH = r'./../../../data/ROIClassification/InputLabelled/*/*/*.*' 
  Recursively traverse folders by adding * in the path'''
  import glob 
  list_of_filename =[]
  for name in glob.glob(PATH, recursive= True):
        filename = name.split('/')[-1]
        list_of_filename.append(filename.split('.')[0]) # just filename without extension 
  return list_of_filename

def traverse_folders_recursive(INPUTPATH):
  for (root,dirs,files) in os.walk(INPUTPATH): 
      for dir in dirs:
          dir_path = os.path.join(root,dir)
          for f in os.listdir(dir_path):
            #add whats to be done
            do_something


def run_condition_per_folder(PATH):
  for f in os.listdir(PATH):
        filepath = os.path.join(PATH,f)
        img = cv2.imread(filepath)
        #add a condition function call
        if condition:
            # add what's to be done
            do_something


def make_new_dir(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)

def check_size_n_save(w,h,OUTPUTPATH,size_thres):
    if (w<size_thres)|(h<size_thres):
        return os.path.join(OUTPUTPATH,'Big')
    else:
        return os.path.join(OUTPUTPATH,'Small')

def exp_lr_schedule(epoch, lr):
    import tensorflow as tf
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5


def save_df_with_header(df,header):
  '''header = ['Actual','Predicted']'''
  df.to_csv('confusionmatrix.csv', header=header)

def save_lists_as_a_df():
  dict = {'Actual':label_batch, 'Pred': predictions}
  df = pd.DataFrame(dict)

def tf_test_on_batch(test_dir, img_height, img_width, batch_size):
  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  image_size=(img_height, img_width),
  batch_size=batch_size)

  image_batch, label_batch = test_ds.as_numpy_iterator().next()
  predictions = model.predict_on_batch(image_batch)#.flatten()

  # Apply a sigmoid since our model returns logits
  predictions = tf.nn.softmax(predictions)
  predictions = np.argmax(predictions, axis =1)
  print('Predictions are:', predictions)

def save_multiple_as_json(acc, loss, val_acc, val_loss):
  '''All inputs are lists of same size'''
  with open(plots_file, 'w') as fd:
    json.dump({'Training': [{
            'Accuracy': a,
            'Loss': l,
            'ValAccuracy': va,
            'ValLoss': vl,
        } for a, l, va, vl in zip(acc, loss, val_acc, val_loss)
    ]}, fd)

  
def plot_side_by_side_image(img, masked_image):
  fig, axs = plt.subplots(1, 2)
  axs[0].imshow(img)
  axs[1].imshow(masked_image)
  plt.show()
    
def plot_side_by_side(x, plt_1, plt_2):
  fig, axs = plt.subplots(1, 2)
  axs[0].plot(x, plt_1)
  axs[1].plot(x, plt_2)
  plt.show()

def plot_side_by_side_3(x, plt_1, plt_2, plt_3):
    fig, axs = plt.subplots(1, 3)
    axs[0].plot(x, plt_1)
    axs[1].plot(x, plt_2)
    axs[2].plot(x, plt_3)
    plt.show()


def crop_image(image,xmin,xmax,ymin,ymax):
    cropped = image[ymin:ymax,xmin:xmax]
    return cropped

def find_match_folder_recursive(string_to_match):
    return [name for name in glob.glob(string_to_match, recursive= True)]


def check_folder_exists(path):
    os.path.exists(path)

def open_grayscale(f):
    gray = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    return gray

def open_image_rgb(f):
    gray = cv2.imread(f)
    return gray

def merge_dfs_left(df1, df2, based_on):
    merged_df = pd.merge(df1, df2, how= "left", on = based_on).reset_index(drop = True)
    return merged_df

def subset_df_per_column(df, column_to_search, list_to_select):
        df_subset = df[df[column_to_search].isin(list_to_select)]
        return df_subset
