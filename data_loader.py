# data_loader.py
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

def load_cifar10(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)
    
    return trainloader, testloader