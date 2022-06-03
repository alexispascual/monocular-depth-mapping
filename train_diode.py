import os
import tensorflow as tf
import pandas as pd

from utils import visualizer
from datasets.diode import DataGenerator
from models.unet import DepthEstimationModel
tf.random.set_seed(123)


def pretrain_on_diode(model: tf.keras.Model,
                      image_height: int,
                      image_width: int,
                      epochs: int,
                      batch_size: float):
    
    annotation_folder = "./diode"

    if not os.path.exists(annotation_folder):

        print("Downloading dataset...")
        os.makedirs(annotation_folder)

        _ = tf.keras.utils.get_file(
            os.path.join(annotation_folder, "val.tar.gz"),
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )

    path = "./diode/val/indoors"

    filelist = []

    for root, _, files in os.walk(path):
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

    train_loader = DataGenerator(data=df[:260].reset_index(drop="true"), 
                                 batch_size=batch_size, 
                                 image_height=image_height,
                                 image_width=image_width)

    validation_loader = DataGenerator(data=df[260:].reset_index(drop="true"), 
                                      batch_size=batch_size, 
                                      image_height=image_height,
                                      image_width=image_width)

    model.fit(train_loader,
              epochs=epochs,
              validation_data=validation_loader)

    return model


def run():
    annotation_folder = "./diode"

    if not os.path.exists(annotation_folder):

        print("Downloading dataset...")
        os.makedirs(annotation_folder)

        _ = tf.keras.utils.get_file(
            os.path.join(annotation_folder, "val.tar.gz"),
            cache_subdir=os.path.abspath("."),
            origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
            extract=True,
        )

    path = "./diode/val/indoors"

    filelist = []

    for root, _, files in os.walk(path):
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

    visualize_samples = next(iter(DataGenerator(data=df, 
                                                batch_size=6, 
                                                image_height=HEIGHT,
                                                image_width=WIDTH)))

    visualizer.visualize_depth_map(visualize_samples)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR,
                                         amsgrad=False)
    model = DepthEstimationModel(width=WIDTH, height=HEIGHT)

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Compile the model
    model.compile(optimizer, loss=cross_entropy)

    train_loader = DataGenerator(data=df[:260].reset_index(drop="true"), 
                                 batch_size=BATCH_SIZE, 
                                 image_height=HEIGHT,
                                 image_width=WIDTH)

    validation_loader = DataGenerator(data=df[260:].reset_index(drop="true"), 
                                      batch_size=BATCH_SIZE, 
                                      image_height=HEIGHT,
                                      image_width=WIDTH)

    model.fit(train_loader,
              epochs=EPOCHS,
              validation_data=validation_loader)

    test_loader = next(iter(DataGenerator(data=df[265:].reset_index(drop="true"), 
                                          batch_size=6, 
                                          image_height=HEIGHT,
                                          image_width=WIDTH)))

    visualizer.visualize_depth_map(test_loader, test=True, model=model)

    test_loader = next(iter(DataGenerator(data=df[300:].reset_index(drop="true"), 
                                          batch_size=6, 
                                          image_height=HEIGHT,
                                          image_width=WIDTH)))

    visualizer.visualize_depth_map(test_loader, test=True, model=model)


if __name__ == '__main__':
    run()
