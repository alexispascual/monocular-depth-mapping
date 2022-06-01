import tensorflow as tf
from utils import tools

from datasets.moonyard import MoonYardDataset
from models.unet import DepthEstimationModel

tf.random.set_seed(123)


def main():
    tf.keras.backend.clear_session()
    
    default_config = './config/training_config.yaml'

    config = tools.load_config(default_config)

    dataset_parameters = config['dataset_parameters']
    experiment_parameters = config['experiment_parameters']

    learning_rate = experiment_parameters['learning_rate']
    epochs = experiment_parameters['epochs']

    image_width = dataset_parameters['image_width']
    image_height = dataset_parameters['image_height']

    dataset = MoonYardDataset(**dataset_parameters)
    dataset.prepare()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         amsgrad=False)

    model = DepthEstimationModel(width=image_width, height=image_height)

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    # Compile the model
    model.compile(optimizer, loss=cross_entropy)

    model.fit(dataset.train_dataset,
              epochs=epochs,
              validation_data=dataset.test_dataset)

    model.save(experiment_parameters['saved_model_dir'])


if __name__ == '__main__':
    main()
