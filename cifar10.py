from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.se_densenet_full_in_loop import se_densenet121
from core.baseline import densenet121
from utils import Trainer


def get_dataloader(batch_size, root="data/cifar10"):
    root = Path(root).expanduser()
    if not root.exists():
        root.mkdir()
    root = str(root)

    to_normalized_tensor = [transforms.ToTensor(),
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    data_augmentation = [transforms.RandomHorizontalFlip(),]

    train_loader = DataLoader(
        datasets.CIFAR10(root, train=True, download=True,
                         transform=transforms.Compose(data_augmentation + to_normalized_tensor)),
        batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        datasets.CIFAR10(root, train=False, transform=transforms.Compose(to_normalized_tensor)),
        batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def main(batch_size, baseline, reduction):
    train_loader, test_loader = get_dataloader(batch_size)

    if baseline:
        model = densenet121()
    else:
        model = se_densenet121(num_classes=10)

    optimizer = optim.SGD(params=model.parameters(), lr=1e-1, momentum=0.9,
                          weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 80, 0.1)
    trainer = Trainer(model, optimizer, F.cross_entropy, save_dir="weights")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--batchsize", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main(args.batchsize, args.baseline, args.reduction)
