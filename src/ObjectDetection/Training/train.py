from absl import app, flags, logging
from absl.flags import FLAGS
#import argparse

import yaml
import os
import sys
#sys.path.append(os.path.join(os.getcwd(),'models/yolov3'))

import tensorflow as tf
import numpy as np
import cv2
import mlflow
mlflow.tensorflow.autolog()
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

# parser = argparse.ArgumentParser()
# parser.add_argument('--paramspath','-m', default= 'params.yaml')
# args = parser.parse_args()

flags.DEFINE_string('paramspath', '', 'path to params file')
flags.DEFINE_string('stage_no', '', 'training stage number')


def main(_argv):
    print(FLAGS.stage_no)
    if FLAGS.stage_no=="1":
        print("Stage 1")
        params = yaml.safe_load(open(FLAGS.paramspath))['Train1']
    if FLAGS.stage_no=="2":
        print("Stage 2")
        params = yaml.safe_load(open(FLAGS.paramspath))['Train2']

    path_dataset = params['path_dataset']
    train_dataset_path=os.path.join(path_dataset,"train.tfrecord")
    validation_dataset_path=os.path.join(path_dataset,"val.tfrecord")

    tiny = params['tiny']
    weights = params['weights']
    classes = params['classes']
    mode = params['mode']
    transfer = params['transfer']
    size = params['size']
    epochs = params['epochs']
    optimizer = params['optimizer']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    num_classes = params['num_classes']
    weights_num_classes = None #params['Train']['weights_num_classes']
    output_checkpoint = params['output_checkpoint']

    if not os.path.exists(output_checkpoint):
        os.makedirs(output_checkpoint)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if tiny:
        model = YoloV3Tiny(size, training=True,
                           classes=num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(size, training=True, classes=num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if train_dataset_path:
        train_dataset = dataset.load_tfrecord_dataset(
            train_dataset_path, classes, size)
    train_dataset = train_dataset.shuffle(buffer_size=512)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if validation_dataset_path:
        val_dataset = dataset.load_tfrecord_dataset(
            validation_dataset_path, classes, size)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, size),
        dataset.transform_targets(y, anchors, anchor_masks, size)))

    # Configure the model for transfer learning
    if transfer == 'none':
        pass  # Nothing to do
    elif transfer in ['darknet', 'no_output']:
        # Darknet transfer is a special case that works
        # with incompatible number of classes

        # reset top layers
        if tiny:
            model_pretrained = YoloV3Tiny(
                size, training=True, classes=weights_num_classes or num_classes)
        else:
            model_pretrained = YoloV3(
                size, training=True, classes=weights_num_classes or num_classes)
        model_pretrained.load_weights(weights)

        if transfer == 'darknet':
            model.get_layer('yolo_darknet').set_weights(
                model_pretrained.get_layer('yolo_darknet').get_weights())
            freeze_all(model.get_layer('yolo_darknet'))

        elif transfer == 'no_output':
            for l in model.layers:
                if not l.name.startswith('yolo_output'):
                    l.set_weights(model_pretrained.get_layer(
                        l.name).get_weights())
                    freeze_all(l)

    else:
        # All other transfer require matching classes
        model.load_weights(weights)

        if transfer == 'fine_tune_everything':
            print("No layer frozen")
        if transfer == 'fine_tune':
            # freeze darknet and fine tune other layers
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)

        elif transfer == 'frozen':
            # freeze everything
            freeze_all(model)
    if optimizer=='adam':
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    if optimizer=='rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate= learning_rate)
    if optimizer=='sgd':
        tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD", **kwargs)
    loss = [YoloLoss(anchors[mask], classes=num_classes)
            for mask in anchor_masks]

    if mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                logging.info("{}_train_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                logging.info("{}_val_{}, {}, {}".format(
                    epoch, batch, total_loss.numpy(),
                    list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(
                os.path.join(output_checkpoint,'checkpoints/yolov3_train.tf'))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(mode == 'eager_fit'))
        # EarlyStopping(patience=3, verbose=1),
        callbacks = [
            ReduceLROnPlateau(verbose=1),            
            ModelCheckpoint(os.path.join(output_checkpoint,'checkpoints/yolov3_train.tf'),
                            verbose=1, save_weights_only=True,save_best_only=True),
            TensorBoard(log_dir=os.path.join(output_checkpoint,'logs'))
        ]

        history = model.fit(train_dataset,
                            epochs=epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


 
