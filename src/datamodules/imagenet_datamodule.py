import lightning as L
from pathlib import Path
from typing import Union
import splitfolders
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import extract_archive

class ImageNetDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size

    # def prepare_data(self):
    #     extract_archive(
    #         from_path="dog-breed-image-dataset.zip",
    #         to_path=self._dl_path,
    #         remove_finished=False
    #     )
    #     splitfolders.ratio(
    #         Path(self._dl_path).joinpath('dataset'), 
    #         output="data/dogs_filtered",
    #         ratio=(.8, .1, .1)
    #     )

    @property
    def data_path(self):
        return Path(self._dl_path).joinpath("imagenet-dataset")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Resize(size=256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("val"), self.valid_transform)
        return DataLoader(
            dataset=dataset, 
            batch_size=self._batch_size, 
            num_workers=self._num_workers, 
            shuffle=train
        )

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False) 