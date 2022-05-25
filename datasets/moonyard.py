import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from .base_dataset import BaseDataset


class MoonYardDataset(BaseDataset):

    """Train and testing dataset for the MoonYard Depth maps
    """

    def __init__(self,
                 root_folder: str,
                 masks_folder: str,
                 image_height: int,
                 image_width: int,
                 channels: int,
                 batch_size: int
                 ):
        super(BaseDataset, self).__init__()

        self.root_folder = root_folder
        self.masks_folder = masks_folder
        self.image_height = image_height
        self.image_width = image_width

        self._channels = channels
        self._batch_size = batch_size

        self.image_file_paths = []
        self.mask_file_paths = []
        self.depth_file_paths = []
        self.point_cloud_file_paths = []

        tqdm.write("Generating file lists...")
        for _, dirs, _ in os.walk(root_folder):
            for directory in dirs:
                self.image_file_paths.append(os.path.join(self.root_folder, directory, f'zed_image_left_{directory}.jpg'))
                self.depth_file_paths.append(os.path.join(self.root_folder, directory, f'depth_map_{directory}.npy'))
                self.point_cloud_file_paths.append(os.path.join(self.root_folder, directory, f'point_cloud_{directory}.npy'))

        for _, _, files in os.walk(masks_folder):
            for f in files:
                self.mask_file_paths.append(os.path.join(self.masks_folder, f))

        tqdm.write("Done!")

    def train_generator(self):
        for image_file, depth_file, mask_file in zip(self.image_file_paths, self.depth_file_paths, self.mask_file_paths):
            image = cv2.imread(image_file)
            mask = cv2.imread(mask_file)
            depth = np.load(depth_file)

            yield image, depth

    def generate_train_dataset(self):

        # return tf.data.Dataset.from_generator(self.train_generator,
        #                                       output_signature=(tf.TensorSpec(shape=(self.image_height, 
        #                                                                              self.image_width,
        #                                                                              self._channels),
        #                                                                       dtype=tf.float32),
        #                                                         tf.TensorSpec(shape=(self.image_height, 
        #                                                                              self.image_width,
        #                                                                              1),
        #                                                                       dtype=tf.float32))
        #                                       ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        return tf.data.Dataset.from_generator(self.train_generator,
                                              output_types=(tf.float32, tf.float32)
                                              ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

    def prepare(self):
        self._image_dataset = self.generate_train_dataset()

    @property
    def train_dataset(self):
        return self._image_dataset
    

if __name__ == '__main__':

    test_config = {'root_folder': './data/moon_yard',
                   'masks_folder': './data/horizons',
                   'image_height': 1080,
                   'image_width': 1920,
                   'channels': 3,
                   'batch_size': 4
                   }

    dataset = MoonYardDataset(**test_config)
    dataset.prepare()

    for image, depth_map in dataset.train_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'])
        print("Train assertion success!")
        break
