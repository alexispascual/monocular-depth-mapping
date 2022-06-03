import os
import train_model
import cv2

import numpy as np
import tensorflow as tf

from utils import tools, view_depth


def main():
    
    default_config = './config/inference_config.yaml'

    config = tools.load_config(default_config)

    directories = config['directories']
    image_parameters = config['image_parameters']
    experiment_parameters = config['experiment_parameters']

    model_dir = directories['model_directory']
    test_dir = directories['test_directory']

    image_height = image_parameters['image_height']
    image_width = image_parameters['image_width']

    show_depth_map_gui = experiment_parameters['show_depth_map_gui']
    visualize_normalized_depth = experiment_parameters['visualize_normalized_depth']

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

            image = tf.squeeze(image, axis=0)
            depth_map = tf.squeeze(depth_map).numpy()
            depth_map = np.exp(depth_map)

            normalized_depth_map = 255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
            normalized_depth_map = normalized_depth_map.astype(np.uint8)
            normalized_depth_map = cv2.cvtColor(normalized_depth_map, cv2.COLOR_GRAY2RGB)

            image = 255 * image.numpy()
            image = image.astype(np.uint8)

            if visualize_normalized_depth:
                tools.show_image(image, normalized_depth_map)

            if show_depth_map_gui:
                view_depth.display_depth(image, depth_map, depth_map)

    cv2. destroyAllWindows()


if __name__ == '__main__':
    main()
