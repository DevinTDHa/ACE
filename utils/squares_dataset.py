from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import csv

import lightning as L


class SquaresDataSet(ImageFolder):
    def __init__(self, label="ColorA", *args, **kwargs):
        """Loads the square data folder. Expected format:

        root
        ├── square.csv
        ├── 0
        │   ├── 0.png
        │   ├── ...
        """
        super(SquaresDataSet, self).__init__(*args, **kwargs)

        self.label_dict = {}
        csv_path = self.root + "/square.csv"
        with open(csv_path, mode="r") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                self.label_dict[row["Name"]] = float(row[label])

    def __getitem__(self, index):
        original_tuple = super(SquaresDataSet, self).__getitem__(index)
        path = self.imgs[index][0]
        base_name = path.split("/")[-1]
        label = torch.Tensor([self.label_dict[base_name]])
        
        dynamic_noise = False
        if (dynamic_noise):
            noise = torch.randn_like(original_tuple[0]) * 0.1
            tuple_with_path = (index, original_tuple[0] + noise, label)
        else:
            tuple_with_path = (index, original_tuple[0], label)
        return tuple_with_path


class SquaresDataModule(L.LightningDataModule):
    def __init__(self, folder_path, csv_path, transform, batch_size=32):
        super(SquaresDataModule, self).__init__()
        self.folder_path = folder_path
        self.csv_path = csv_path
        self.transform = transform
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = SquaresDataSet(self.folder_path, transform=self.transform)
        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [0.8, 0.2]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)
