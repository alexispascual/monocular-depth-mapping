import tensorflow as tf
import os

from datasets.diode import DiodeDataset


def pretrain_on_diode(model: tf.keras.Model,
                      **kwargs):
    
    dataset = DiodeDataset(**kwargs)
    dataset.prepare()

    checkpoint_file = kwargs['checkpoint_file']
    epochs = kwargs['epochs']

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file,
                                                             save_weights_only=True,
                                                             verbose=1)

    checkpoint_dir = os.path.dirname(kwargs['checkpoint_file'])

    if os.path.isdir(checkpoint_dir) and len(checkpoint_dir) != 0:
        print("Loading model from checkpoint...")
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        checkpoint_epoch = int(tf.train.latest_checkpoint(checkpoint_dir).split("-")[1].split(".")[0])
        print(f"Found checkpoint trained for {checkpoint_epoch} epochs!")
        epochs -= checkpoint_epoch

    model.fit(dataset.train_dataset,
              epochs=epochs,
              validation_data=dataset.val_dataset,
              callbacks=[checkpoint_callback])

    return model


if __name__ == '__main__':
    pass
