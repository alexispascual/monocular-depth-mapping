import tensorflow as tf
import numpy as np
import cv2
import os

from .base_dataset import BaseDataset
from utils import tools, view_depth


class DiodeDataset(BaseDataset):
    def __init__(self, 
                 root_folder: str = './data/diode',
                 train_folder: str = './data/diode/train/outdoors',
                 val_folder: str = './data/diode/val/outdoors',
                 image_height: int = 768, 
                 image_width: int = 1024, 
                 image_channels: int = 3,
                 depth_channels: int = 1,
                 batch_size: int = 6,
                 shuffle: bool = True,
                 **kwargs):
        """
        Initialization
        """

        if not os.path.exists(root_folder):

            print("Downloading dataset...")
            os.makedirs(root_folder)

            _ = tf.keras.utils.get_file(
                os.path.join(root_folder, "val.tar.gz"),
                cache_subdir=os.path.abspath("."),
                origin="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
                extract=True)
        else:
            print(f"Diode dataset found in {root_folder}")

        self.train_data = self.generate_filelist(train_folder)
        self.val_data = self.generate_filelist(val_folder)
        self.image_height = image_height
        self.image_width = image_width
        self.depth_channels = depth_channels

        self._image_channels = image_channels
        self._batch_size = batch_size

        self.max_depth = 300
        self.min_depth = 0.1

        num_train_files = len(self.train_data["image"])
        num_val_files = len(self.val_data["image"])

        assert num_train_files != 0, "No training files found!"
        assert num_val_files != 0, "No validation files found!"

        self.train_indices = list(range(num_train_files))
        self.val_indices = list(range(num_val_files))

        print(f"{num_train_files} training images found.")
        print(f"{num_val_files} validation images found.")

        if shuffle:
            np.random.shuffle(self.train_indices)
            np.random.shuffle(self.val_indices)

    def load(self, image_path, depth_map, mask):
        """Load input and target image."""

        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_width, self.image_height))
        image = tf.image.convert_image_dtype(image, tf.float32)

        depth_map = np.load(depth_map).squeeze()

        mask = np.load(mask)
        mask = mask > 0
        
        max_depth = min(300, np.percentile(depth_map, 99))

        if max_depth == 0:
            max_depth = self.max_depth

        depth_map = np.clip(depth_map, self.min_depth, max_depth)
        depth_map = np.log(depth_map, where=mask)

        depth_map = np.ma.masked_where(~mask, depth_map)

        depth_map = np.clip(depth_map, 0.1, np.log(max_depth))
        depth_map = cv2.resize(depth_map, (self.image_width, self.image_height))
        depth_map = np.expand_dims(depth_map, axis=2)
        depth_map = tf.image.convert_image_dtype(depth_map, tf.float32)

        return image, depth_map

    def train_generator(self):
        for i in self.train_indices:
            image, depth_map = self.load(self.train_data['image'][i],
                                         self.train_data['depth'][i],
                                         self.train_data['mask'][i])

            yield image, depth_map

    def val_generator(self):
        for i in self.val_indices:
            image, depth_map = self.load(self.val_data['image'][i],
                                         self.val_data['depth'][i],
                                         self.val_data['mask'][i])

            yield image, depth_map

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

    def generate_val_dataset(self):
        """
        output_tpyes and output_shapes will be deprecated in a future version but pyright thinks
        TensorSpec does not accept arguments.
        
        """
        return tf.data.Dataset.from_generator(self.val_generator,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=((self.image_height, self.image_width, self.image_channels), 
                                                             (self.image_height, self.image_width, self.depth_channels))
                                              ).batch(self._batch_size).prefetch(tf.data.AUTOTUNE)    

    def generate_filelist(self, folder):
        filelist = []

        for root, _, files in os.walk(folder):
            for file in files:
                filelist.append(os.path.join(root, file))

        filelist.sort()
        data = {
            "image": [x for x in filelist if x.endswith(".png")],
            "depth": [x for x in filelist if x.endswith("_depth.npy")],
            "mask": [x for x in filelist if x.endswith("_depth_mask.npy")],
        }

        return data

    def prepare(self):
        self._train_dataset = self.generate_train_dataset()
        self._val_dataset = self.generate_val_dataset()

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def image_channels(self):
        return self._image_channels


if __name__ == '__main__':

    test_config = {'root_folder': './data/diode',
                   'train_folder': './data/diode/train/outdoor',
                   'val_folder': './data/diode/val/outdoor',
                   'image_height': 768,
                   'image_width': 1024,
                   'image_channels': 3,
                   'depth_channels': 1,
                   'batch_size': 4,
                   'shuffle': True
                   } 

    dataset = DiodeDataset(**test_config)
    dataset.prepare()

    for image, depth_map in dataset.train_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['image_channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['depth_channels'])

        depth_map_ = np.exp(depth_map[0].numpy())
        normalized_depth_map = 255 * (depth_map_ - np.min(depth_map_)) / (np.max(depth_map_) - np.min(depth_map_))
        normalized_image = 255 * image[0].numpy()

        tools.show_image(normalized_image.astype(np.uint8), cv2.cvtColor(normalized_depth_map.astype(np.uint8), cv2.COLOR_GRAY2RGB))
        view_depth.display_depth(normalized_image.astype(np.uint8), np.squeeze(depth_map_), np.squeeze(depth_map_))

        print("Train dataset assertion success!")
        break

    for image, depth_map in dataset.val_dataset:
        assert image.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['image_channels'])
        assert depth_map.shape == (test_config['batch_size'], test_config['image_height'], test_config['image_width'], test_config['depth_channels'])
        print("Validation dataset assertion success!")
        break
