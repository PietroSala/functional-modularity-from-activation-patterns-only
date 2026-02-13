
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import torch
from torch import nn

from dataprovider.dataprovider import BaseDataProvider


class FashionMnistDataProvider(BaseDataProvider):
    def __init__(self, 
                 batch_size=32,
                 autoencoder=False,
                 shuffle=True,
                 split=[0.7,0.3],
                 device=None,
                 cnn = False,
                 num_hidden=32,
                 num_kernels=32
                 ):
        super().__init__(
            batch_size=batch_size, 
            split=split,
            device=device
        )
        self.name = 'mnist_fashion_cnn' 
        if autoencoder: self.name + '_auto'

        self.autoencoder = autoencoder
        self.num_features = 784
        self.num_hidden = num_hidden
        self.num_kernels = num_kernels
        

        self.dataset = FashionMnistDataset()
        self.dataset_train, self.dataset_val = self.dataset.random_split(split)
        self.dataset_test = FashionMnistDataset(train=False)
        
        self.train = DataLoader(self.dataset_train, batch_size, shuffle )#, num_workers=1)
        self.val = DataLoader(self.dataset_val, batch_size, shuffle )#, num_workers=1)
        self.test = DataLoader(self.dataset_test, batch_size, shuffle )#, num_workers=1)

        if self.autoencoder:
            self.num_classes = self.num_features
            self.model = FashionMnistAutoencoderModel(self.num_features, self.num_hidden)
            self.criterion = nn.MSELoss()
        else:
            self.num_classes = 10
            if cnn:
                self.model = FashionMnistCNNModel(self.num_kernels)
            else:
                self.model = FashionMnistModel(self.num_features, self.num_hidden, self.num_classes) #Type: nn.Module        

            self.criterion = nn.CrossEntropyLoss()

        self.model = self.model.to(self.device)

class FashionMnistModel(nn.Module):
    def __init__(self, input_size, num_hidden, num_classes):
        super(FashionMnistModel, self).__init__()
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, num_hidden)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(num_hidden, num_hidden) 
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden, num_hidden) 
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.flat(x)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        x = self.softmax(x)
        return x



class FashionMnistCNNModel(nn.Module):
    
    def __init__(self, num_kernels=32):
        super(FashionMnistCNNModel, self).__init__()
        self.num_kernels = num_kernels

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=num_kernels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_kernels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=num_kernels, out_channels=num_kernels*2, kernel_size=3),
            nn.BatchNorm2d(num_kernels*2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=(num_kernels*2)*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        #out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

class FashionMnistAutoencoderModel(nn.Module):
    def __init__(self, input_size, num_hidden):
        super(FashionMnistAutoencoderModel, self).__init__()
        w = h = int(input_size ** 0.5)
        self.w = w
        self.h = h
        num_hidden2 = num_hidden // 2
        num_hidden4 = num_hidden // 4
        
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, num_hidden)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(num_hidden, num_hidden2)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden2, num_hidden4)
        self.act3 = nn.ReLU()
        self.layer4 = nn.Linear(num_hidden4, num_hidden2)
        self.act4 = nn.ReLU()
        self.layer5 = nn.Linear(num_hidden2, num_hidden)
        self.act5 = nn.ReLU()
        self.layer6 = nn.Linear(num_hidden, input_size)
        self.act6 = nn.ReLU()

    def forward(self, x):
        x = self.flat(x)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.layer4(x)
        x = self.act4(x)
        x = self.layer5(x)
        x = self.act5(x)
        x = self.layer6(x)
        x = self.act6(x)
        x = x.reshape(-1, self.w, self.h)
        return x


class FashionMnistDataset(Dataset):
    
    def __init__(self, train=True, device=None, indices=None):
        self.train=train
        self.indices=indices
        self.ds = datasets.FashionMNIST('./data', train=train, download=True)
        self.name_class = self.ds.classes
        self.num_class = len(self.name_class)
        
        self.x = self.ds.data.to(dtype=torch.float) / 255
        self.y = self.ds.targets

        if indices is not None:
            self.x = self.x[indices]
            self.y = self.y[indices]

        #y = torch.nn.functional.one_hot(y,self.num_class)
        #self.y = y.to(dtype=torch.float)
        
        
        if device is None: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = device
        
    def copy(self, indices=None):
        return FashionMnistDataset(self.train, self.device, indices=indices)

    def random_split(self, split):
        subsets = random_split(self, split)
        ds_subs = []
        for subset in subsets:
            ds_sub = self.copy(subset.indices)
            ds_subs.append(ds_sub)
        return ds_subs

        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        #return self.x[idx].to(self.device), self.y[idx].to(self.device)
    
