import os
import cv2
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle
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
                 batch_size: int,
                 train_test_split: float
                 ):
        super(BaseDataset, self).__init__()

        self.root_folder = root_folder
        self.masks_folder = masks_folder
        self.image_height = image_height
        self.image_width = image_width

        self._channels = channels
        self._batch_size = batch_size

        # self.image_file_paths = []
        # self.mask_file_paths = []
        # self.depth_file_paths = []
        # self.point_cloud_file_paths = []
        self.folder_names = []

        tqdm.write("Generating file list...")
        for _, dirs, _ in os.walk(root_folder):
            for directory in dirs:
                self.folder_names.append(directory)

        tqdm.write("Done!")
        tqdm.write(f"{len(self.folder_names)} files found.")

        self.folder_names = shuffle(self.folder_names)

        train_split = math.floor(train_test_split * len(self.folder_names))
        self.train_folders = self.folder_names[:train_split]
        self.test_folders = self.folder_names[train_split:]

        tqdm.write(f"{len(self.train_folders)} training files...")
        tqdm.write(f"{len(self.test_folders)} testing files...")

    def train_generator(self):

        for dirname in self.train_folders:
            image_path = os.path.join(self.root_folder, dirname, f'zed_image_left_{dirname}.jpg')
            depth_file_path = os.path.join(self.root_folder, dirname, f'depth_map_{dirname}.npy')
            # pount_cloud_file = os.path.join(self.root_folder, dirname, f'point_cloud_{dirname}.npy')

            mask_file = os.path.join(self.masks_folder, f'horizon_{dirname}.jpg')

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_file)
            depth = np.load(depth_file_path)

            yield image, depth

    def test_generator(self):

        for dirname in self.test_folders:
            image_path = os.path.join(self.root_folder, dirname, f'zed_image_left_{dirname}.jpg')
            depth_file_path = os.path.join(self.root_folder, dirname, f'depth_map_{dirname}.npy')
            # pount_cloud_file = os.path.join(self.root_folder, dirname, f'point_cloud_{dirname}.npy')
            
            mask_file = os.path.join(self.masks_folder, f'horizon_{dirname}.jpg')

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_file)
            depth = np.load(depth_file_path)

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

    def generate_test_dataset(self):
        return tf.data.Dataset.from_generator(self.train_generator,
                                              output_types=(tf.float32, tf.float32)
                                              ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

    def prepare(self):
        self._train_dataset = self.generate_train_dataset()
        self._test_dataset = self.generate_test_dataset()

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


if __name__ == '__main__':

    test_config = {'root_folder': './data/moon_yard',
                   'masks_folder': './data/horizons',
                   'image_height': 1080,
                   'image_width': 1920,
                   'channels': 3,
                   'batch_size': 4,
                   'train_test_split': 0.75
                   }

    dataset = MoonYardDataset(**test_config)
    dataset.prepare()

    for image, depth_map in dataset.train_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'])
        print("Train assertion success!")
        break

    for image, depth_map in dataset.test_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'])
        print("Test assertion success!")
        break
