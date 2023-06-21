from torch.utils.data import DataLoader
import pytorch_lightning as pl
import albumentations as A
from albumentations.core.composition import Compose
from albumentations.pytorch import ToTensorV2
from dataset import Dataset
import pathlib
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from augmentation.cutout import Cutout

class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_dir='train', 
        test_dir='test', 
        image_size=512, 
        n_valid=0, 
        n_splits=5, 
        batch_size=32, 
        num_workers=4, 
        aug=None,
        p=0
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.image_size = image_size
        self.n_valid = n_valid
        self.n_splits = n_splits
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug = aug
        self.p = p
        self.train_df = self.process(self.train_dir, train=True)
        self.test_df = self.process(self.test_dir, train=False)
        self.train_transforms, self.valid_transform = self.init_transforms()

    def init_transforms(self):
        train_transforms = [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]

        if self.aug == 'cutout':
            train_transforms.insert(0, Cutout(num_holes=1, max_h_size=128, max_w_size=128, p=self.p))

        train_transforms = Compose(train_transforms)

        valid_transform = Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        return train_transforms, valid_transform

    def setup(self, stage=None):
        train_df = self.train_df[self.train_df.fold != self.n_valid].reset_index(drop=True)
        val_df = self.train_df[self.train_df.fold == self.n_valid].reset_index(drop=True)

        self.train_dataset = Dataset(train_df, image_size=self.image_size, transforms=self.train_transforms)
        self.valid_dataset = Dataset(val_df, image_size=self.image_size, transforms=self.valid_transform)

        self.test_dataset = Dataset(self.test_df, image_size=self.image_size, transforms=self.train_transforms, train=False)

    def train_dataloader(self):
        return self.dataloader(self.train_dataset, train=True)

    def val_dataloader(self):
        return self.dataloader(self.valid_dataset)

    def test_dataloader(self):
        return self.dataloader(self.test_dataset)

    def dataloader(self, dataset: Dataset, train: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=train,
        )
    
    def process(self, data_dir, train=True):
        if train:
            path=pathlib.Path(data_dir)
            filepaths=list(path.glob(r"*/*.jpg"))
            labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))
            df1=pd.Series(filepaths,name='file_path').astype(str)
            df2=pd.Series(labels,name='label')
            df=pd.concat([df1,df2],axis=1)

            df['image_id'] = df['file_path'].apply(lambda image:image.split('/')[-1])

            encoder = LabelEncoder()
            label2index = {l: i for (i, l) in enumerate(encoder.fit(df["label"]).classes_)}
            index2label = {x[1]: x[0] for x in label2index.items()}

            df["label_index"] = encoder.fit_transform(df["label"])

            skf = StratifiedKFold(n_splits=self.n_splits)
            for fold, (_, val_) in enumerate(skf.split(X=df, y=df.label)):
                df.loc[val_, "fold"] = fold

            return df

        else:
            path=pathlib.Path(data_dir)
            filepaths=list(path.glob("*.jpg"))
            labels=list(map(lambda x: os.path.split(os.path.split(x)[0])[1],filepaths))
            df1=pd.Series(filepaths,name='file_path').astype(str)
            df2=pd.Series(labels,name='label')
            df=pd.concat([df1,df2],axis=1)

            df['image_id'] = df['file_path'].apply(lambda image:image.split('/')[-1])
            df['label'] = -1
            df["label_index"] = -1

            return df
