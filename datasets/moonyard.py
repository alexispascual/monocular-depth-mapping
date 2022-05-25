import os

from .base_dataset import BaseDataset


class MoonYardDataset(BaseDataset):

    """Train and testing dataset for the MoonYard Depth maps
    """

    def __init__(self,
                 root_folder: str,
                 masks_folder: str
                 ):
        super(BaseDataset, self).__init__()

        self.root_folder = root_folder
        self.masks_folder = masks_folder

        self.image_file_paths = []
        self.mask_file_paths = []
        self.depth_file_paths = []
        self.point_cloud_file_paths = []

        for _, dirs, _ in os.walk(root_folder):
            for directory in dirs:
                self.image_file_paths.append(os.path.join(self.root_folder, directory, f'zed_image_left_{directory}.jpg'))
                self.depth_file_paths.append(os.path.join(self.root_folder, directory, f'depth_map_{directory}.jpg'))
                self.point_cloud_file_paths.append(os.path.join(self.root_folder, directory, f'point_cloud_{directory}.jpg'))

        for _, _, files in os.walk(masks_folder):
            for f in files:
                self.mask_file_paths.append(os.path.join(self.masks_folder, f))

        print(self.image_file_paths)
        print(self.depth_file_paths)
        print(self.mask_file_paths)

    def prepare(self):
        pass


if __name__ == '__main__':

    test_config = {'root_folder': './data/moon_yard',
                   'masks_folder': './data/horizons'}

    dataset = MoonYardDataset(**test_config)
