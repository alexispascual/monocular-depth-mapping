class BaseDataset(object):
    """Default Superclass for Dataset classes"""

    def __init__(self):
        super(BaseDataset, self).__init__()

    def prepare(self):
        raise NotImplementedError('A dataset prepare method must be implemented!')

    @property
    def train_dataset(self):
        raise NotImplementedError('train_dataset property must be overwritten!')

    @property
    def test_dataset(self):
        raise NotImplementedError('test_dataset property must be overwritten!')

    @property
    def batch_size(self):
        raise NotImplementedError('batch_size property must be overwritten!')

    @property
    def channels(self):
        raise NotImplementedError('channels property must be overwritten!')

    @property
    def num_classes(self):
        raise NotImplementedError('num_classes property must be overwritten!')

    @property
    def batches(self):
        raise NotImplementedError('batches property must be overwritten!')
