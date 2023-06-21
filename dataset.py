import torch
import cv2

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df, 
        image_size=512, 
        transforms=None, 
        train=True
    ):
        super().__init__()
        self.image_size = image_size
        self.transforms = transforms
        self.train = train

        self.image_path = df['file_path'].values
        self.labels = df["label_index"].values
        self.ids = df['image_id']

    def __len__(self):
        return len(self.image_path)

    def resize(self, image, interp):
        return  cv2.resize(image, (self.image_size, self.image_size), interpolation=interp)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image_id = self.ids[index]

        image_path = self.image_path[index]
        image = cv2.imread(image_path, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.resize(image, cv2.INTER_AREA)

        if self.transforms is not None:
            result = self.transforms(image=image)
            image = result['image']

        if self.train:
            return {'image':image, 'target': int(label)}
        else:
            return {'image':image, 'target': image_id}


