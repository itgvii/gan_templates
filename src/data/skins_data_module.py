import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader  # PIL-loader из torchvision


class SingleClassDatasetFolder(DatasetFolder):
    def find_classes(self, directory):
        # Override to treat all files as a single class
        return ['.'], {'.': 0}


class SkinsDataModule(L.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 batch_size: int = 16,
                 train_loader_num_worker: int = 4,
                 val_loader_num_worker: int = 4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.train_loader_num_worker = train_loader_num_worker
        self.val_loader_num_worker = val_loader_num_worker
        self.data_report = {}
        self.model_hyperparams = {}

    def setup(self, stage: str):
            
        if stage == 'train':
            
            transform = transforms.Compose([
                transforms.ToTensor(),               # перевод в Tensor [0,1],
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            dataset = SingleClassDatasetFolder(
                root=self.data_path,              # здесь лежат все .jpg/.png
                loader=default_loader,            # используем стандартный PIL-loader
                extensions=('png'),               # какие расширения читать
                transform=transform,              # применять трансформации
                target_transform=None,            # нам не нужны метки
            )

            self.train_dataset, self.val_dataset = random_split(dataset, lengths=[0.3, 0.7])


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.train_loader_num_worker,
            pin_memory=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.val_loader_num_worker,
            pin_memory=True
            )


if __name__ == '__main__':
    dm = SkinsDataModule('tmp', file_path='/home/yan/projects/minecraft/data/Skins')
    dm.setup('train')
    batch = next(iter(dm.train_dataloader()))
    print(batch.shape)