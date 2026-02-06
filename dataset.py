import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_dataloaders(data_dir, batch_size=32):

    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform_train)

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(0.7*n)
    val_end = int(0.85*n)

    train_ds = Subset(dataset, indices[:train_end])
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=transform_test),
                    indices[train_end:val_end])
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=transform_test),
                     indices[val_end:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, len(dataset.classes)
