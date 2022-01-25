import os
import sys

import tensorflow as tf
from tensorflow.keras import layers

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import vizualizer
from datasets.diode import DataGenerator
from models.unet import DepthEstimationModel
tf.random.set_seed(123)

def main():
    annotation_folder = "/downloads/"

    if not os.path.exists(os.path.abspath(".") + annotation_folder):
        annotation_zip = tf.keras.utils.get_file(
            "val.tar.gz",
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )

    path = "./downloads/val/indoors"

    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    filelist.sort()
    data = {
        "image": [x for x in filelist if x.endswith(".png")],
        "depth": [x for x in filelist if x.endswith("_depth.npy")],
        "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
    }
    df = pd.DataFrame(data)

    df = df.sample(frac=1, random_state=42)

    HEIGHT = 256
    WIDTH = 256
    LR = 0.0002
    EPOCHS = 30
    BATCH_SIZE = 32

    visualize_samples = next(iter(DataGenerator(data=df, batch_size=6, dim=(HEIGHT, WIDTH))))

    vizualizer.visualize_depth_map(visualize_samples)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR,
                                         amsgrad=False)
    model = DepthEstimationModel()

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Compile the model
    model.compile(optimizer, loss=cross_entropy)

    train_loader = DataGenerator(data=df[:260].reset_index(drop="true"), 
                                 batch_size=BATCH_SIZE, 
                                 dim=(HEIGHT, WIDTH))

    validation_loader = DataGenerator(data=df[260:].reset_index(drop="true"), 
                                      batch_size=BATCH_SIZE, 
                                      dim=(HEIGHT, WIDTH))

    model.fit(train_loader,
              epochs=EPOCHS,
              validation_data=validation_loader)

    test_loader = next(iter(DataGenerator(data=df[265:].reset_index(drop="true"), 
                                          batch_size=6, 
                                          dim=(HEIGHT, WIDTH))))

    visualize_depth_map(test_loader, test=True, model=model)

    test_loader = next(iter(DataGenerator(data=df[300:].reset_index(drop="true"), 
                                          batch_size=6, 
                                          dim=(HEIGHT, WIDTH))))

    visualize_depth_map(test_loader, test=True, model=model)



if __name__ == '__main__':
    main()