import tensorflow as tf

from datasets.diode import DiodeDataset


def pretrain_on_diode(model: tf.keras.Model,
                      **kwargs):
    
    dataset = DiodeDataset(**kwargs)
    dataset.prepare()

    model.fit(dataset.train_dataset,
              epochs=kwargs['epochs'],
              validation_data=dataset.test_dataset)

    return model


if __name__ == '__main__':
    pass
