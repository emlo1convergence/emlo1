from resnet import ResNet18
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 15
batch_size = 128
learning_rate = 0.1

transform = transforms.Compose([
                                transforms.RandomResizedCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# CIFAR10 dataset 
train_dataset = datasets.CIFAR10(root='./data', 
                                           train=True, 
                                           transform=transform,  
                                           download=True)

test_dataset = datasets.CIFAR10(root='./data', 
                                          train=False, 
                                          transform=transform)

# Data loader
train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

model = ResNet18().to(device)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data,target) in enumerate(pbar):
        data,target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.nll_loss(y_pred, target)
        loss.backward()
        optimizer.step()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id = {batch_idx} Accuracy = {100*correct/processed:0.2f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    best_test_accuracy = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100*correct/len(test_loader.dataset)))

torch.save(model.state_dict(), "cifar_resnet18.pth")

model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
EPOCHS = num_epochs

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

