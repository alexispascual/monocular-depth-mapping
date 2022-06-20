import tensorflow as tf
import os

from datasets.diode import DiodeDataset


def pretrain_on_diode(model: tf.keras.Model,
                      **kwargs):
    
    dataset = DiodeDataset(**kwargs)
    dataset.prepare()

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=kwargs['checkpoint_file'],
                                                             save_weights_only=True,
                                                             verbose=1)

    checkpoint_dir = os.path.dirname(kwargs['checkpoint_file'])
    
    if len(checkpoint_dir) != 0:
        model.load_weights(kwargs['checkpoint_file'])

    model.fit(dataset.train_dataset,
              epochs=kwargs['epochs'],
              validation_data=dataset.val_dataset,
              callbacks=[checkpoint_callback])

    return model


if __name__ == '__main__':
    pass
