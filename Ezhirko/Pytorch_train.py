from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary
from tqdm import tqdm
import os
import sys
import pandas as pd

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)
traindir = os.path.join('data', 'train')
validdir = os.path.join('data', 'validation')
batch_size = 10
train_losses = []
test_losses = []
train_acc = []
test_acc = []
cat_acc = []
dog_acc = []
epchs=[]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

simple_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.4859, 0.4470, 0.4086],[0.2559, 0.2475, 0.2476])
                                        ])
# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root=traindir, transform=simple_transforms),
    'valid':
    datasets.ImageFolder(root=validdir, transform=simple_transforms),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
}

# executed to calculate mean
#channels_sum, channels_squared_sum, num_batches = 0,0,0
#for data,_ in dataloaders['train']:
#    channels_sum += torch.mean(data, dim=[0,2,3])
#    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
#    num_batches +=1
    
#mean = channels_sum/num_batches
#mean = channels_sum/num_batches
#std = (channels_squared_sum/num_batches - mean**2)**0.5

#print('mean:',mean)
#print('std:',std)
#mean: tensor([0.4859, 0.4470, 0.4086])
#std: tensor([0.2559, 0.2475, 0.2476])

model = models.vgg16(pretrained=True)
#summary(model,input_size=(1,224,224))
#print(model)

n_inputs = model.classifier[6].in_features
# Freeze model weights
for param in model.parameters():
    param.requires_grad = False

#Add classification layer
model.classifier[6] = nn.Sequential(
                      nn.Linear(n_inputs,256),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(256,2),
                      nn.LogSoftmax(dim=1))
                      
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data,target) in enumerate(pbar):
        data,target = data.to(device), target.to(device)
        #target = target.to(torch.float32)
        optimizer.zero_grad()
        y_pred = model(data)
        #y_pred = torch.squeeze(y_pred,dim=1)
        #loss = F.binary_cross_entropy(y_pred, target)
        loss = F.nll_loss(y_pred, target)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        pbar.set_description(desc=f'Loss={loss.item()} Batch_id = {batch_idx} Accuracy = {100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    count_0 = 0
    count_1 = 0
    correct_0 = 0
    correct_1 = 0
    best_test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #target = target.to(torch.float32)
            output = model(data)
            #output = torch.squeeze(output,dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            pred_num = pred.cpu().numpy()
            labels = target.cpu().numpy()
            for label, predict in zip(labels, pred_num):
                if(label == 0):
                    count_0 += 1
                    if(label == predict[0]):
                        correct_0 += 1
                else:
                    count_1 += 1
                    if(label == predict[0]):
                        correct_1 += 1
            
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Class0 Accuracy: ({:.2f}%), Class1 Accuracy: ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100*correct/len(test_loader.dataset),(100*(correct_0/count_0)),(100*(correct_1/count_1))))
    
    test_acc.append(100*correct/len(test_loader.dataset))
    cat_acc.append(100*(correct_0/count_0))
    dog_acc.append(100*(correct_1/count_1))
    if test_acc[-1] > best_test_accuracy:
      best_test_accuracy = test_acc[-1]
      save_model(model)
    
def save_model(model):
    torch.save(model.state_dict(),'best-model-parameters.pt')
    
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 11

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    epchs.append(epoch)
    train(model, device, dataloaders['train'], optimizer, epoch)
    test(model, device, dataloaders['val'])

metrics = []
for eph,train_ls,train_ac,val_ls,val_ac,cat_ac,dog_ac in zip(epchs,train_losses,train_acc,test_losses,test_acc,cat_acc,dog_acc):
    mtx = [eph,train_ls,train_ac,val_ls,val_ac,cat_ac,dog_ac]
    metrics.append(mtx)
    
clmns = ['Epoch','TrainLoss','TrainAccuracy','TestLoss','TestAccuracy','CatAccuracy','DogAccuracy']
metrics_df = pd.DataFrame(metrics,columns=clmns)
metrics_df.to_csv('Metrics.csv',index=False)


    

