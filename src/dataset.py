import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path

class ChestCTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.dataset = datasets.ImageFolder(data_dir, transform=transform)
        self.transform = transform
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

        self.images = []
        self.labels = []
        for i in range(len(self.dataset)):
            self.images.append(self.dataset.samples[i][0])
            self.labels.append(self.dataset.samples[i][1])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def get_dataloaders(data_dir, batch_size=16, image_size=224, val_split=0.2, test_dir=None):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = full_dataset.classes

    test_loader = None
    if test_dir and Path(test_dir).exists():
        test_dataset = ChestCTDataset(test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, classes