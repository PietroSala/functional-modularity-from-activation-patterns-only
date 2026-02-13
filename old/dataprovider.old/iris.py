from sklearn import datasets
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import os

from dataprovider.dataprovider import BaseDataProvider


class IrisDataProvider(BaseDataProvider):
    def __init__(self, 
                 batch_size=32,
                 shuffle=True,
                 num_hidden=20,
                 split=[1,0]):
        super().__init__(batch_size=batch_size, )
        self.name = 'iris'
        self.num_features = 4
        
        self.criterion = nn.CrossEntropyLoss()
        self.dataset = IrisDataset()

        self.train = DataLoader(self.dataset,batch_size,shuffle,num_workers=1)
        self.val = DataLoader(self.dataset,batch_size,shuffle,num_workers=1)
        self.model = IrisModel(self.num_features, num_hidden, self.dataset.num_class)

class IrisModel(nn.Module):
    def __init__(self, input_size, num_hidden, num_classes):
        super(IrisModel, self).__init__()
        self.layer1 = nn.Linear(input_size, num_hidden)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(num_hidden, num_hidden)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x


class IrisDataset(Dataset):
    
    def __init__(self):
        self.ds = iris = datasets.load_iris()
        self.name_class = iris['target_names']
        self.num_class = len(self.name_class)
        
        self.x = torch.FloatTensor( iris['data'] )
        y = torch.LongTensor(iris['target'])
        self.y = y
        #y = torch.nn.functional.one_hot(y,self.num_class)
        #self.y = y.to(dtype=torch.float)
        


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
