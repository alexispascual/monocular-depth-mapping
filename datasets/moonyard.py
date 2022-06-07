import os
import cv2
import math
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle
from .base_dataset import BaseDataset
from utils import tools, view_depth


class MoonYardDataset(BaseDataset):

    """Train and testing dataset for the MoonYard Depth maps
    """

    def __init__(self,
                 root_folder: str,
                 masks_folder: str,
                 image_height: int,
                 image_width: int,
                 image_channels: int,
                 depth_channels: int,
                 batch_size: int,
                 train_test_split: float
                 ):
        super(BaseDataset, self).__init__()

        self.root_folder = root_folder
        self.masks_folder = masks_folder
        self.image_height = image_height
        self.image_width = image_width
        self.depth_channels = depth_channels

        self._image_channels = image_channels
        self._batch_size = batch_size

        self.min_depth = 0.2
        self.max_depth = 20
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

            image = self.image_transforms(image, mask)
            depth = self.mask_depth_map(depth, mask)

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

            image = self.image_transforms(image, mask)
            depth = self.mask_depth_map(depth, mask)

            yield image, depth

    def generate_train_dataset(self):
        """
        output_tpyes and output_shapes will be deprecated in a future version but pyright thinks
        TensorSpec does not accept arguments.

        Use output_signature in the future
        
        """
        
        # return tf.data.Dataset.from_generator(self.train_generator,
        #                                       output_signature=(tf.TensorSpec(shape=(self.image_height, 
        #                                                                              self.image_width,
        #                                                                              self._image_channels),
        #                                                                       dtype=tf.float32),
        #                                                         tf.TensorSpec(shape=(self.image_height, 
        #                                                                              self.image_width,
        #                                                                              1),
        #                                                                       dtype=tf.float32))
        #                                       ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        return tf.data.Dataset.from_generator(self.train_generator,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((self.image_height, self.image_width, self.image_channels), 
                                                             (self.image_height, self.image_width, self.depth_channels))
                                              ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

    def generate_test_dataset(self):
        """
        output_tpyes and output_shapes will be deprecated in a future version but pyright thinks
        TensorSpec does not accept arguments.
        
        """
        return tf.data.Dataset.from_generator(self.train_generator,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((self.image_height, self.image_width, self.image_channels), 
                                                             (self.image_height, self.image_width, self.depth_channels))
                                              ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

    def mask_depth_map(self, _depth_map, _mask):
        mask = cv2.resize(_mask, (_depth_map.shape[1], _depth_map.shape[0]))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask == 255

        invalid_mask = np.isnan(_depth_map)

        depth_map = np.nan_to_num(_depth_map)
        depth_map = np.where(mask, depth_map, self.max_depth)
        depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
        depth_map = np.log(depth_map, where=~invalid_mask)

        depth_map = np.ma.masked_where(invalid_mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(self.max_depth))
        depth_map = cv2.resize(depth_map, (self.image_width, self.image_height))

        depth_map = np.expand_dims(depth_map, axis=2)

        return depth_map

    def prepare(self):
        self._train_dataset = self.generate_train_dataset()
        self._test_dataset = self.generate_test_dataset()

    def image_transforms(self, image, mask):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = mask == 255

        image = np.where(mask, image, (0, 0, 0))

        image = tf.image.resize(image, (self.image_height, self.image_width))
        image = tf.image.convert_image_dtype(image, tf.float32)

        return image

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def image_channels(self):
        return self._image_channels


if __name__ == '__main__':

    test_config = {'root_folder': './data/moon_yard',
                   'masks_folder': './data/horizons',
                   'image_height': 768,
                   'image_width': 1024,
                   'image_channels': 3,
                   'depth_channels': 1,
                   'batch_size': 4,
                   'train_test_split': 0.75
                   } 

    dataset = MoonYardDataset(**test_config)
    dataset.prepare()

    for image, depth_map in dataset.train_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['image_channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['depth_channels'])

        depth_map_ = np.exp(depth_map[0].numpy())
        normalized_depth_map = 255 * (depth_map_ - np.min(depth_map_)) / (np.max(depth_map_) - np.min(depth_map_))

        tools.show_image(image[0].numpy().astype(np.uint8), cv2.cvtColor(normalized_depth_map.astype(np.uint8), cv2.COLOR_GRAY2RGB))
        view_depth.display_depth(image[0].numpy().astype(np.uint8), np.squeeze(depth_map_), np.squeeze(depth_map_))

        print("Train assertion success!")
        break

    for image, depth_map in dataset.test_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['image_channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['depth_channels'])
        print("Test assertion success!")
        break
