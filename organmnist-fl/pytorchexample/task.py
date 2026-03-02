"""pytorchexample: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, ToTensor
import medmnist
from medmnist import OrganAMNIST, INFO


# Información del dataset
info = INFO['organamnist']
NUM_CLASSES = len(info['label'])  # 11 clases de órganos


class Net(nn.Module):
    """CNN mejorada para OrganAMNIST (28x28 grayscale, 11 clases)"""
    def __init__(self):
        super(Net, self).__init__()
        # Bloque 1
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Bloque 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Bloque 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14→7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7→3
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


class OrganAMNISTDataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.dataset = OrganAMNIST(split=split, download=True, size=28)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return {
            "image": img,
            "label": int(label.item()) if hasattr(label, 'item') else int(label[0])
        }


# Cache para particiones
_partition_cache = {}


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int,
    partitioner_type: str = "iid",
    alpha: float = 0.5,
    num_classes_per_partition: int = 2,
):
    """Load partition OrganAMNIST data."""
    global _partition_cache

    cache_key = (num_partitions, partitioner_type, alpha, num_classes_per_partition)

    if cache_key not in _partition_cache:
        full_dataset = OrganAMNISTDataset(split="train", transform=pytorch_transforms)
        # Obtener etiquetas sin aplicar transforms
        raw_dataset = OrganAMNIST(split="train", download=True, size=28)
        labels = np.array([int(raw_dataset[i][1].item()) for i in range(len(raw_dataset))])

        if partitioner_type == "iid":
            indices_per_partition = _iid_partition(labels, num_partitions)
        elif partitioner_type == "dirichlet":
            indices_per_partition = _dirichlet_partition(labels, num_partitions, alpha, NUM_CLASSES)
        elif partitioner_type == "pathological":
            indices_per_partition = _pathological_partition(labels, num_partitions, num_classes_per_partition, NUM_CLASSES)
        else:
            raise ValueError(f"Particionador desconocido: {partitioner_type}")

        _partition_cache[cache_key] = (full_dataset, indices_per_partition)

    full_dataset, indices_per_partition = _partition_cache[cache_key]
    indices = indices_per_partition[partition_id]

    split = int(0.8 * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]

    trainloader = DataLoader(
        torch.utils.data.Subset(full_dataset, train_indices),
        batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(
        torch.utils.data.Subset(full_dataset, test_indices),
        batch_size=batch_size
    )
    return trainloader, testloader


def _iid_partition(labels, num_partitions):
    indices = np.random.permutation(len(labels))
    return np.array_split(indices, num_partitions)


def _dirichlet_partition(labels, num_partitions, alpha, num_classes):
    indices_per_class = [np.where(labels == c)[0] for c in range(num_classes)]
    partition_indices = [[] for _ in range(num_partitions)]
    for class_indices in indices_per_class:
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = (proportions * len(class_indices)).astype(int)
        proportions[-1] = len(class_indices) - proportions[:-1].sum()
        start = 0
        for p, count in enumerate(proportions):
            partition_indices[p].extend(class_indices[start:start+count].tolist())
            start += count
    return [np.array(p) for p in partition_indices]


def _pathological_partition(labels, num_partitions, num_classes_per_partition, num_classes):
    all_classes = list(range(num_classes))
    partition_indices = [[] for _ in range(num_partitions)]
    for p in range(num_partitions):
        assigned_classes = all_classes[
            (p * num_classes_per_partition) % num_classes:
            (p * num_classes_per_partition) % num_classes + num_classes_per_partition
        ]
        if len(assigned_classes) < num_classes_per_partition:
            assigned_classes += all_classes[:num_classes_per_partition - len(assigned_classes)]
        for c in assigned_classes:
            class_indices = np.where(labels == c)[0]
            partition_indices[p].extend(class_indices.tolist())
    return [np.array(p) for p in partition_indices]


def load_centralized_dataset():
    """Load test set and return dataloader."""
    test_dataset = OrganAMNISTDataset(split="test", transform=pytorch_transforms)
    return DataLoader(test_dataset, batch_size=128)


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy