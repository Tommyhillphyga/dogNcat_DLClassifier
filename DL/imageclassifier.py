import os
from sched import scheduler
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torchvision import models
import shutil
import torch
import torchvision
import torch.nn as nn
import cv2
import time
from torch import optim, tensor
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random
import time 


path = 'datasets/train'
main_folder = os.listdir(path)

files = []
cats = []
dogs = []
for file in main_folder:
    sub_folder= os.listdir(os.path.join(path, file))
    for ani in sub_folder:
        full_path = os.path.join(path, file, ani)
        files.append(full_path)

for pet in files:
    label = 'cat'
    target = pet.split('\\')[-1].split('.')[0]
    if label == target:
        cats.append(pet)
    if label != target:
        dogs.append(pet)

# sample_test = random.sample(files, 2000)
# dest_path = 'datasets/test'
# for file in sample_test:
#     shutil.move(file, dest_path)

# print('test dataset moved successfully')

simple_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_path = 'datasets/train'  
valid_path = 'datasets/valid'
test_path = 'datasets/test'

train = ImageFolder(train_path, simple_transform)
valid = ImageFolder(valid_path, simple_transform)
test = ImageFolder(test_path, simple_transform)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=True)

dataloaders = {'train': train_loader, 'valid': valid_loader, 'test': test}
dataset_size = {'train': len(train), 'valid': len(valid), 'test': len(test)}

#building network architecture
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ft = model.fc.in_features
model.fc = nn.Linear(num_ft, 2)

#building loss and optimization method

criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer_ft = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.9)
exp_scheduler = lr_scheduler.StepLR(optimizer= optimizer_ft, step_size=7, gamma= 0.1)

def TrainModel(model, criterion, schedular, optimizer, num_epoch = 25):
    print( 'model trainig begins....')
    since = time.time()
    best_model = model.state_dict()
    best_accuracy = 0.0

    for epoch in range(num_epoch - 1):
        print('epoch {}/{}'.format(epoch, num_epoch))
        print('-' * 11)

        for phase in ['train', 'valid']:
            running_loss = 0.0
            running_correct = 0

            if phase == 'train':

                model.train(True)
            else:
                model.train(False)

            for data in dataloaders[phase]:
                inputs, labels = data
                inputs = Variable(inputs)
                labels = Variable(labels)

                optimizer.zero_grad()
                #forward
                outputs =  model(inputs)
                _,predicted = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                #backward + optimizer
                if phase == 'train':

                    loss.backward()
                    optimizer.step()
                    schedular.step()

                #print statistics
                running_loss += loss
                running_correct += torch.sum(predicted == labels)
            
            epoch_loss = running_loss / dataset_size[phase]
            epoch_accuracy = running_correct / dataset_size[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))

                #deep copy model
            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = model.state_dict()

    time_elaspsed = time.time() - since

    print('Training complete in, {:.0f}m {:.0f}s'.format( time_elaspsed//60, time_elaspsed%60))
    print('Best model accuracy {:.4f}'.format(best_accuracy))

    #load best model weights
    model.load_state_dict(best_model)
    
    #save the model
    # torch.save(best_model, '/content/drive/MyDrive/DL/model.pth')
    return model

# test model on test dataset

def Testmodel():
  print('model testing begins...')
  #load model
  msg = 'test completed'
  device = torch.device('cpu')
  loaded_model = model
  loaded_model.load_state_dict(torch.load('/content/drive/MyDrive/DL/model.pth', map_location=device))
  #loaded_model = loaded_model.to(device)
  loaded_model.eval()

  running_correct = 0
  total = 0
  
  with torch.no_grad():
    for data in dataloaders['test']:
      inputs, labels = data
      inputs = Variable(inputs)
      labels = Variable(labels)

      outputs = loaded_model(inputs)
      _,predicted = torch.max(outputs.data, 1)
      
      total += labels.size(0)
      running_correct += torch.sum(predicted == labels)

  print(f'Accuracy of the network on the 2000 test images: {100 * running_correct / total} %')
  return msg


if __name__ == '__main__':
    print()
    # TrainModel(model, criterion=criterion, schedular = exp_scheduler, optimizer=optimizer_ft)