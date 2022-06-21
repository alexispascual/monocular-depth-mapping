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
    moonyard_checkpoint_file = experiment_parameters['checkpoint_file']
    moonyard_checkpoint_dir = os.path.dirname(moonyard_checkpoint_file)

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

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=moonyard_checkpoint_file,
                                                             save_weights_only=True,
                                                             verbose=1)

    if tf.train.latest_checkpoint(moonyard_checkpoint_dir) is not None:
        print("Loading model from checkpoint...")
        model.load_weights(tf.train.latest_checkpoint(moonyard_checkpoint_dir))
        checkpoint_epoch = str(tf.train.latest_checkpoint(moonyard_checkpoint_dir))
        checkpoint_epoch = int(checkpoint_epoch.split("-")[1].split(".")[0])
        print(f"Found checkpoint trained for {checkpoint_epoch} epochs!")
    else:
        print("Found no checkpoints! Starting from scratch...")
        checkpoint_epoch = 0

    model.fit(dataset.train_dataset,
              epochs=epochs,
              initial_epoch=checkpoint_epoch,
              validation_data=dataset.test_dataset,
              callbacks=[checkpoint_callback])

    model.save(experiment_parameters['saved_model_dir'])

    return experiment_parameters['saved_model_dir']

    
if __name__ == '__main__':
    main()
