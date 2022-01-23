
#Augmentation
python src/data_processing/RotateImagesWithBB.py --input_csv_path data/annotation/train.csv --output_csv_path data/annotation/train_rotated.csv --input_images_root data/train_Images --output_images_root data/train_image_rotated

#TO create TF Records

python src/data_processing/generate_tfrecord.py --paramspath experiment/OD_LOGO_DETECTOR/params.yaml 

#To Visualize the tfrecord

python src/ObjectDetection/Training/visualize_dataset.py --classes src/ObjectDetection/Training/data/lines.names --dataset experiment/OD_LOGO_DETECTOR/TEMP/tf_record/train.tfrecord

#To train the model 

python src/ObjectDetection/Training/train.py --paramspath experiment/OD_LOGO_DETECTOR/params.yaml  --stage_no 1
python src/ObjectDetection/Training/train.py --paramspath experiment/OD_LOGO_DETECTOR/params.yaml  --stage_no 2

#Detect
python src/ObjectDetection/Training/detect_for_mAP.py --paramspath experiment/OD_LOGO_DETECTOR/params.yaml 

#For mAP
python src/metric/mAP/generate_ground_truth_for_mAP.py

python src/metric/mAP/scripts/extra/intersect-gt-and-dr.py

python src/metric/mAP/main.py

# Export Tfserving
python  src/ObjectDetection/Inference/export_tfserving.py

#Test tfserve model
python src/ObjectDetection/Inference/RunInference.py