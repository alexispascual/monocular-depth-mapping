import os
from pprint import pp
import train_model
import cv2

import numpy as np
import tensorflow as tf

from utils import tools


def main():
    
    default_config = './config/inference_config.yaml'

    config = tools.load_config(default_config)

    directories = config['directories']
    image_parameters = config['image_parameters']

    model_dir = directories['model_directory']
    test_dir = directories['test_directory']

    image_height = image_parameters['image_height']
    image_width = image_parameters['image_width']

    if not os.path.isdir(model_dir):
        print("Model does not exist! Training model...")
        model_dir = train_model.main()

    print(f"Loading model from: {model_dir}")
    model = tf.keras.models.load_model(model_dir)

    model.summary()

    for _, dirs, _ in os.walk(test_dir):
        for directory in dirs:
            image_file = os.path.join(test_dir, directory, f'zed_image_left_{directory}.jpg')
            image = cv2.imread(image_file)
            image = cv2.resize(image, (image_width, image_height))
            image = np.expand_dims(image, axis=0)
            image = tf.image.convert_image_dtype(image, tf.float32)

            depth_map = model.predict(image)


if __name__ == '__main__':
    main()
