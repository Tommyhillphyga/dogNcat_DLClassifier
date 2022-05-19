
from pytest import importorskip
import torch
import cv2 
import numpy as np
import pandas as pd
import os
import calstats as cs
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
from torch import optim, tensor
from torch.optim import lr_scheduler
import time


path = 'petImages'
main_folder = os.listdir(path)
cats = []
dogs= []
for i in range(len(main_folder)):
    sub_folder = os.listdir(os.path.join(path, main_folder[i]))
    #print(sub_folder)
    for file in sub_folder:
        full_path = os.path.join(path, main_folder[i], file)
        if i == 0:
            cats.append(full_path)
        elif i == 1:
            dogs.append(full_path)


cats_dataset = np.array(cats).reshape(-1,1)
dogs_dataset = np.array(dogs).reshape(-1,1)

cats_labels = np.zeros((cats_dataset.shape[0]), dtype= int).reshape(-1, 1)
dogs_labels = np.ones((dogs_dataset.shape[0]), dtype= int).reshape(-1, 1)


cats = np.hstack([cats_dataset, cats_labels])
dogs = np.hstack([dogs_dataset, dogs_labels])

data = np.vstack([cats, dogs])

df = pd.DataFrame(data=data, columns=["img_path", "label"])
df['label'] = df['label'].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(df['img_path'], df.label, test_size=0.2, random_state=42)

xt = np.array(X_train).reshape(-1, 1)
yt = np.array(y_train).reshape(-1, 1)
xv = np.array(X_valid).reshape(-1, 1)
yv = np.array(y_valid).reshape(-1, 1)

df_train = np.hstack([xt, yt])
df_valid = np.hstack([xv, yv])

train_data = pd.DataFrame(df_train, columns=['img_path', 'label'])
valid_data = pd.DataFrame(df_valid, columns=['img_path', 'label'])



from torch.utils.data import DataLoader, Dataset
from PIL import Image as im

class DogsNCats(Dataset):

    def __init__(self, df_, transform=None) -> None:
        self.data = df_
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_path = self.data["img_path"].iloc[idx]
        label = self.data["label"].iloc[idx]
        label = torch.tensor(label)
        img = im.open(img_path).convert('RGB')

        if self.transform != None:
            img = self.transform(img)
            

        return img, label


from torchvision import transforms

simple_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_dataset = DogsNCats(train_data, transform=simple_transform)
valid_dataset = DogsNCats(valid_data, transform=simple_transform)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size= 32, shuffle=False )

#mean_std = cs.getmeanNstd(train_loader)

#building network architecture
model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_ft = model.fc.in_features
model.fc = nn.Linear(num_ft, 2)

#training the model
learning_rate = 0.001
criterion_ = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer_ft, step_size=7, gamma=0.1)

dataloaders = {'train':train_loader, 'valid': valid_loader}
size_ = {'train': len(train_dataset), 'valid': len(valid_dataset)}


#training the model
def TrainModel( model, criterion, scheduler, num_epochs = 25):

    since = time.time()
    best_model = model.state_dict()
    best_accuracy = 0.0

    for epoch in range (num_epochs):
        print('Epoch {}/{}'.format( epoch, num_epochs-1))
        print('-' * 11)

        for phase in ['train', 'valid']:
            if phase == 'train':
                #scheduler.step()
                model.train(True)  #set model to training mode
            else:
                model.train(False)  #set model to evaluation mode
            running_loss = 0.0
            running_correct = 0   
            # loading the data
            for data in dataloaders[phase]:
                inputs, labels = data
                #print('input: =  ', inputs, 'label = ', labels)
                #wrap them into deep learning variable
                inputs = Variable(inputs)
                labels =  Variable(labels)
                
                optimizer_ft.zero_grad()
                outputs = model(inputs)
                _,predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                #backward + optimize only in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer_ft.step()
                    scheduler.step()

                #print statistics
                running_loss += loss
                running_correct += torch.sum(predicted == labels)
            
            epoch_loss = running_loss / size_[phase] 
            epoch_accuracy = running_correct / size_[phase]

            print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_accuracy))



            #print()
            if phase == 'valid' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = model.state_dict()

    time_elaspsed = time.time() - since

    print('Training complete in, {:.0f}m {:.0f}s'.format( time_elaspsed//60, time_elaspsed%60))
    print('Best model accuracy {:.4f}'.format(best_accuracy))

    #load best model weights
    model.load_state_dict(best_model)
    return model






if __name__ == "__main__":
    TrainModel(model, criterion= criterion_, scheduler= exp_lr_scheduler)
    print('classifier loading successful')
    # for data in dataloaders['train']:
    #     inputs, labels = data
    #     print(inputs, labels)

