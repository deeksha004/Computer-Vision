import torch
import numpy as np

#from gluoncv.utils import plot_images
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import random
seed =0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
def _init_fn(worker_id):
    np.random.seed(int(seed))
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("Cuda is not available.Training on CPU..")
else:
    print("Cuda is available.Training on GPU")
import matplotlib.pyplot as plt

label_names = ['with_heads', 'without_heads']

def get_train_valid_loader(data_dir_train,
                           #data_dir_test,
                           batch_size,
                           is_train,
                           augment=False,
                           random_seed = 0,
                           valid_size=0.3,
                           shuffle=True,
                           show_sample=False,
                           num_workers=0,
                           pin_memory=False):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    
    if is_train:

        # define transforms
        valid_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
        ])
        if augment:
            train_transform = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        # load the dataset
        train_dataset = datasets.ImageFolder(
            root=data_dir_train, 
            transform=train_transform
        )

        valid_dataset = datasets.ImageFolder(
            root=data_dir_train, 
            transform=valid_transform
        )
        print(train_dataset.class_to_idx)
        print(valid_dataset.class_to_idx)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=_init_fn
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
            worker_init_fn=_init_fn
        )

        # visualize some images
        if show_sample:
            sample_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=9, shuffle=shuffle,
                num_workers=num_workers, pin_memory=pin_memory,
                worker_init_fn=_init_fn
            )
            data_iter = iter(sample_loader)
            images, labels = data_iter.next()
            X = images.numpy().transpose([0, 2, 3, 1])
            plot_images(X, labels)

        return (train_loader, valid_loader)
    else:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        
data_dir_train = "train_heads"

batch_size = 16
augment = False
train_loader, valid_loader = get_train_valid_loader(data_dir_train,
                                                    #data_dir_test,
                                                   batch_size=batch_size,
                                                    is_train = True,
                                                   augment = False,
                                                   random_seed = 0,
                                                   valid_size=0.3,
                                                   shuffle=True,
                                                   show_sample=False,
                                                   num_workers=0,
                                                   pin_memory=False)
                                                   
def conv(in_channels, out_channels, kernel_size, stride=1, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
    
    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)
class Network(nn.Module):   
    # Making the fc n/w wide
    def __init__(self):
        super(Network,self).__init__()
        
        self.conv1 = conv(3,64,3)
        self.conv2 = conv(64,128,3)
        self.conv3 = conv(128,256,3)
        self.conv4 = conv(256,512,3)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(512*1*1, 16384)
        self.fc2 = nn.Linear(16384,4096)
        self.fc3 = nn.Linear(4096,1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 2)
        self.drop = nn.Dropout(.4)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        #print(x.shape)
        x = x.view(-1,512*1*1)
        x = self.drop(x)
        x = self.drop(F.relu(self.fc1(x)))
        
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        x = self.drop(F.relu(self.fc4(x)))
        x = F.softmax(self.fc5(x))
        return x   
network = Network()
#print(network)
        
if train_on_gpu:
    network.cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(network.parameters(),lr= 0.001, momentum = 0.9)
optimizer = optim.SGD(network.parameters(),lr= 0.001, momentum=0.9, nesterov=True)

import time
print ("Start Execution : ",end="") 
start_time = time.ctime()
print(start_time) 
epochs = 4
val_loss_min = np.Inf
for e in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0
    training_corrects = 0
    valid_corrects = 0
    network.train()
    for images, labels in train_loader:
        # = data
        if train_on_gpu:
            images,labels = images.cuda(),labels.cuda()
        
        optimizer.zero_grad()
        output = network.forward(images)
        _, preds = torch.max(output, 1)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()*images.size(0)
        training_corrects += torch.sum(preds == labels.data)
    
    network.eval()
    for images,labels in valid_loader:
        
        if train_on_gpu:
            images,labels = images.cuda(),labels.cuda()
        
        output = network.forward(images)
        _, preds_valid = torch.max(output, 1)
        loss = criterion(output,labels)
        valid_loss += loss.item()*images.size(0)
        valid_corrects += torch.sum(preds_valid == labels.data)
        
    train_acc = training_corrects.double() / (len(train_loader)*batch_size)
    valid_acc = valid_corrects.double() / (len(valid_loader)*batch_size) 
    training_loss = training_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)
    
    print('Epoch : {} \tTraining Loss : {: .4f} \tTraining accuracy : {: .4f} \tValidation Loss : {: .4f}  \tvalidation accuracy : {: .4f}'.format(e+1,training_loss,train_acc, valid_loss, valid_acc))
    
    if valid_loss <= val_loss_min:
        print('Validation Loss decreased ({: .4f} ---> {: .4f}. Saving Model)'.format(val_loss_min,valid_loss))
        torch.save(network.state_dict(),'model_detail.pt')
        val_loss_min = valid_loss
print ("Stop Execution : ",end="") 
end_time = time.ctime()
print(end_time)
