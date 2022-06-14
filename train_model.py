import os

import tensorflow as tf
from utils import tools

from datasets.moonyard import MoonYardDataset
from models.unet import DepthEstimationModel
from train_diode import pretrain_on_diode

tf.random.set_seed(123)


def main():
    tf.keras.backend.clear_session()
    
    default_config = './config/training_config.yaml'

    config = tools.load_config(default_config)

    dataset_parameters = config['dataset_parameters']
    experiment_parameters = config['experiment_parameters']
    diode_parameters = config['diode_parameters']

    learning_rate = experiment_parameters['learning_rate']
    epochs = experiment_parameters['epochs']
    pretrain = experiment_parameters['pretrain']

    image_width = dataset_parameters['image_width']
    image_height = dataset_parameters['image_height']

    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         amsgrad=False)

    # Define model
    model = DepthEstimationModel(height=image_height, width=image_width)

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Compile the model
    model.compile(optimizer, loss=cross_entropy)

    if pretrain:
        if not os.path.isdir(diode_parameters['saved_model_dir']):
            print("Pre-training model on diode dataset...")
            model = pretrain_on_diode(model=model, 
                                      **diode_parameters)

            model.save(diode_parameters['saved_model_dir'])
        else:
            print("Fetching model pretrained on diode dataset...")
            model = tf.keras.models.load_model(diode_parameters['saved_model_dir']) 

    dataset = MoonYardDataset(**dataset_parameters)
    dataset.prepare()

    model.fit(dataset.train_dataset,
              epochs=epochs,
              validation_data=dataset.test_dataset)

    model.save(experiment_parameters['saved_model_dir'])

    return experiment_parameters['saved_model_dir']

    
if __name__ == '__main__':
    main()
