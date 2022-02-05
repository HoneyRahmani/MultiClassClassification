# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 21:18:14 2021

@author: asus
"""
from torchvision import datasets
import torchvision.transforms as transforms
import os

path2data = "./data"
if not os.path.exists(path2data):
    os.mkdir(path2data)
data_transformer = transforms.Compose([transforms.ToTensor()])
train_ds = datasets.STL10(path2data, split = 'train',
                          download = False,transform=data_transformer)
print(train_ds.data.shape)

#=========
import collections

y_train = [y for _,y in train_ds]
#Count the number of images per category
counter_train = collections.Counter(y_train)
print(counter_train)

#========
test0_ds = datasets.STL10(path2data, split = 'test',
                          download = False,transform=data_transformer)

print(test0_ds.data.shape)

#=====
from sklearn.model_selection import StratifiedShuffleSplit
# StratifiedShuffleSplit is similar to ShuffleSplit,
# but StratifiedShuffleSplit preservs the percentage of samples for each class.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = list(range(len(test0_ds)))
y_test0 = [y for _,y in test0_ds]

for test_index, val_index in sss.split(indices, y_test0):
    print("test:", test_index,"val:",val_index)
    print(len(val_index), len(test_index))
# Define val_ds, test_ds

from torch.utils.data import Subset

val_ds = Subset(test0_ds, val_index)
test_ds = Subset(test0_ds,test_index)


import collections


y_test = [y for _,y in test_ds]
y_val = [y for _,y in val_ds]

count_val = collections.Counter(y_val)
count_test = collections.Counter(y_test)

print("number of val_ds is:",count_val)
print("number of test_ds is:",count_test)


#======Show a sample image in train_ds and val_ds
#===train_ds
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def show(img, y=None, color=True):
    
    #convert PyTorch tensors(img is pytorch tensor) into NumPy arrays
    npimg = img.numpy()
    npimg_tr = np.transpose(npimg, (1,2,0))
    
    plt.imshow(npimg_tr)
    if y is not None:
        plt.title("label:" +str(y))


grid_size = 4
rnd_ids = np.random.randint(0, len(train_ds), grid_size)
print(rnd_ids)

x_grid = [train_ds[i][0] for i in rnd_ids]
y_grid = [train_ds[i][1] for i in rnd_ids]

x_grid = utils.make_grid(x_grid, nrow=4, padding = 1)
print(x_grid.shape)

plt.figure(figsize=(10,10))
show(x_grid, y_grid)

#===val_ds
np.random.seed(0)
grid_size = 4
rnd_ids = np.random.randint(0, len(val_ds), grid_size)
print(rnd_ids)

x_grid = [val_ds[i][0] for i in rnd_ids]
y_grid = [val_ds[i][1] for i in rnd_ids]

x_grid = utils.make_grid(x_grid, nrow=4, padding = 1)
print(x_grid.shape)

plt.figure(figsize=(10,10))
show(x_grid, y_grid)
#========================Preprocessing 
#===Calcute mean and standard deviation

import numpy as np

mean_RGB = [np.mean(x.numpy(), axis = (1,2)) for x, _ in train_ds]
std_RGB = [np.std(x.numpy(), axis = (1,2)) for x, _ in train_ds]

meanR = np.mean([m[0] for m in mean_RGB])
meanG = np.mean([m[1] for m in mean_RGB])
meanB = np.mean([m[2] for m in mean_RGB])

stdR = np.std([m[0] for m in mean_RGB])
stdG = np.std([m[1] for m in mean_RGB])
stdB = np.std([m[2] for m in mean_RGB])

print(meanR,meanG,meanB)
print(stdR,stdG,stdB)

# =======Define the image transformations in train_ds and test_ds
#train_ds

train_transformer =  transforms.Compose([ transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize([meanR,meanG,meanB], [stdR,stdG,stdB])])

test_transformer =  transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([meanR,meanG,meanB], [stdR,stdG,stdB])])

train_ds.transform = train_transformer
test0_ds.transform = test_transformer

#=============Show samples from trasforms datasets
import torch

np.random.seed(0)
torch.manual_seed(0)

grid_size = 4
rnd_ids = np.random.randint(0, len(train_ds), grid_size)
print(rnd_ids)

x_grid = [train_ds[i][0] for i in rnd_ids]
y_grid = [train_ds[i][1] for i in rnd_ids]

x_grid = utils.make_grid(x_grid, nrow=4, padding = 2)
print(x_grid.shape)

plt.figure(figsize=(10,10))
show(x_grid, y_grid)
#==============================Create DataLoader
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

for x, y in train_dl:
    
    print(x.shape)
    print(y.shape)
    break
    
for x, y in val_dl:
    
    print(x.shape)
    print(y.shape)
    break


########################################################################################################
###############Building models#########################
##Randomly weight
from torchvision import models
import torch


model_resnet18 = models.resnet18(pretrained=False)
print(model_resnet18)
### change number of class in output layer
from torch import nn
num_classes = 10
num_ftrs = model_resnet18.fc.in_features
model_resnet18.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0")
model_resnet18.to(device)

from torchsummary import summary
summary(model_resnet18, input_size=(3,224,224),device=device.type)

#############get the weights
for w in model_resnet18.parameters():
    w=w.data.cpu()
    print(w.shape)
    break
#normalize weights
min_w = torch.min(w)
w1 = (-1/(2*min_w))*w+0.5
print(torch.min(w1).item(),torch.max(w1).item())

#make grid
grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)

plt.figure(figsize=(10,10))
show(x_grid)

##pretrained weight
from torchvision import models
import torch


resnet18_pretrained = models.resnet18(pretrained=True)
# To know layers in resnet18, spicefically last layer
print(resnet18_pretrained)
### change number of class in output layer
from torch import nn
num_classes = 10
#Return number of input fetures
num_ftrs = resnet18_pretrained.fc.in_features
#"fc" is name of last layer, "(fc): Linear(in_features=512, out_features=1000, bias=True)"
resnet18_pretrained.fc = nn.Linear(num_ftrs, num_classes)
device = torch.device("cuda:0")
resnet18_pretrained.to(device)


from torchsummary import summary
summary(resnet18_pretrained, input_size=(3,224,224),device=device.type)

# #############get the weights
for w in resnet18_pretrained.parameters():
     w=w.data.cpu()
     print(w.shape)
     break
##normalize weights
min_w = torch.min(w)
w1 = (-1/(2*min_w))*w+0.5
print(torch.min(w1).item(),torch.max(w1).item())
# 
##make grid
grid_size = len(w1)
x_grid = [w1[i] for i in range(grid_size)]
x_grid = utils.make_grid(x_grid, nrow=8, padding=1)
print(x_grid.shape)
# 
plt.figure(figsize=(10,10))
show(x_grid)
############################Defenation LossFunction
# =============================================================================
# loss_func = nn.CrossEntropyLoss(reduction="sum")
ls_f = nn.LogSoftmax(dim=1)
# ############################Define the optimizer and the learning rate schedule
# ###Define the optimizer
from torch import optim
# opt = optim.Adam(model_resnet18.parameters(),lr=1e-4)
# ###Define the learning rate schedule
def get_lr(opt):
     for param_groups in opt.param_groups:
         return param_groups['lr']
# current_lr = get_lr(opt)
# print('current lr:{}'.format(current_lr))
# ###Define the learning rate schedule
from torch.optim.lr_scheduler import CosineAnnealingLR
# lr_scheduler = CosineAnnealingLR(opt, T_max=2, eta_min=1e-5)
# =============================================================================
# =============================================================================
# for i in range(10):
#     lr_scheduler.step()
#     print("epoch %s, lr:%.1e"%(i,get_lr(opt)))
# =============================================================================
###############################################################
#######Train and Transfer learning
def metrics_batch (output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects
#######compute the loss value per batch of data
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metrics_batch(output, target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item(), metric_b

#######compute the loss value and the performance metric for the entire dataset or an epoch
def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)
        loss_b, metric_b = loss_batch(loss_func, output, yb,opt)
        running_loss+=loss_b
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
        
    loss = running_loss/float(len_data)
    metric = running_metric/float(len_data)
    return loss, metric
######## Train-val Function
import copy
def train_val(model,params):
    
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    
    loss_history = {
        "train": [],
        "val": [],
        }
    
    metric_history = {
        "train": [],
        "val": [],
        }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current_lr={}'.format(epoch,num_epochs-1,
                                                  current_lr))
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl,sanity_check)
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
            
        # save the best model parameters
        if val_loss < best_loss:
                
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied the best model weights!")
        lr_scheduler.step()
        print ("train loss: %.6f, val loss: %.6f, accuracy: %.2f"
                   %(train_loss, val_loss, 100*val_metric))
        print("-"*10)
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history

# Call the train_val function
import copy

loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model_resnet18.parameters(),lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt, T_max=5, eta_min = 1e-6)

os.makedirs("./models", exist_ok=True)

params_train = {
    "num_epochs": 10,
    "optimizer" : opt,
    "loss_func" : loss_func,
    "train_dl" : train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler" : lr_scheduler,
    "path2weights" : "./models/resnet18.pt",   
    }

model_resnet18, loss_hist, metric_hist= train_val(model_resnet18, params_train)


num_epochs = params_train["num_epochs"]
plt.title("Train-Val Acuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label = "Train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

num_epochs = params_train["num_epochs"]
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label = "Train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.xlabel("Training Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

###############################

import copy

loss_func = nn.CrossEntropyLoss(reduction="sum")
opt = optim.Adam(model_resnet18.parameters(),lr=1e-4)
lr_scheduler = CosineAnnealingLR(opt, T_max=5, eta_min = 1e-6)

os.makedirs("./models", exist_ok=True)

params_train = {
    "num_epochs": 10,
    "optimizer" : opt,
    "loss_func" : loss_func,
    "train_dl" : train_dl,
    "val_dl": val_dl,
    "sanity_check": False,
    "lr_scheduler" : lr_scheduler,
    "path2weights" : "./models/resnet18_pretrained.pt",   
    }

model_resnet18, loss_hist, metric_hist= train_val(resnet18_pretrained, params_train) 
           
            
    
    
    
        
        

    


