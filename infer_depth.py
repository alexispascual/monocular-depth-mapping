import os
import train_model

import tensorflow as tf

from utils import tools


def main():
    
    default_config = './config/inference_config.yaml'

    config = tools.load_config(default_config)

    directories = config['directories']

    model_dir = directories['model_directory']

    if not os.path.isdir(model_dir):
        print("Model does not exist! Training model...")
        model_dir = train_model.main()

    print(f"Loading model from: {model_dir}")
    model = tf.keras.models.load_model(model_dir)

    model.summary()


if __name__ == '__main__':
    main()
