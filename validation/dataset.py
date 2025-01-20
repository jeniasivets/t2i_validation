import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io


class ImageTextDataset(Dataset):
    def __init__(self, csv_file, image_dir, transforms=None):
        """
        csv_file: path to csv file
        image_dir: directory with images
        """
        super().__init__()
        self.dataframe = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_name = self.dataframe.iloc[idx]['url'].split('/')[4] + '.png'
        image_path = os.path.join(self.image_dir, image_name)
        image = io.imread(image_path)
        if self.transforms:
            image = self.transforms(image)
        text = self.dataframe.iloc[idx]['caption']
        sample = {'image': image, 'text': text, 'image_path': image_path, 'url': self.dataframe.iloc[idx]['url']}
        return sample
