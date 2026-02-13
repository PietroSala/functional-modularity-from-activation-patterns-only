
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
import torch
from torch import nn

from dataprovider.dataprovider import BaseDataProvider


class MnistDataProvider(BaseDataProvider):
    def __init__(self, 
                 batch_size=32,
                 shuffle=True,
                 split=[0.7,0.3],
                 device=None
                 ):
        super().__init__(
            batch_size=batch_size, 
            split=split,
            device=device
        )
        self.name = 'mnist_mlp'

        self.num_features = 784
        self.num_hidden = 64
        self.num_classes = 10
        self.criterion = nn.CrossEntropyLoss()

        self.dataset = MnistDataset()
        self.dataset_train, self.dataset_val = self.dataset.random_split(split)
        self.dataset_test = MnistDataset(train=False)
        
        self.train = DataLoader(self.dataset_train, batch_size, shuffle )#, num_workers=1)
        self.val = DataLoader(self.dataset_val, batch_size, shuffle )#, num_workers=1)
        self.test = DataLoader(self.dataset_test, batch_size, shuffle )#, num_workers=1)

        #self.model = MnistModelSpase(self.num_features, self.num_hidden, self.num_classes) #Type: nn.Module
        
        self.model = MnistModel(self.num_features, self.num_hidden, self.num_classes) #Type: nn.Module
        self.model = self.model.to(self.device)

class MnistModel(nn.Module):
    def __init__(self, input_size, num_hidden, num_classes):
        super(MnistModel, self).__init__()
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, num_hidden)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(num_hidden, num_hidden)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flat(x)
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.softmax(x)
        return x

class MnistModelSpase(nn.Module):
    def __init__(self, input_size, num_hidden, num_classes, l1_lambda=0.01):
        super(MnistModelSpase, self).__init__()
        self.l1_lambda = l1_lambda
        
        self.flat = nn.Flatten()
        self.layer1 = nn.Linear(input_size, num_hidden)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(num_hidden, num_hidden)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(num_hidden, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        # Store activations for L1 regularization
        self.activations = []

    def add_activations(self, x):
        self.activations.append(x)
    
    def clear_activations(self):
        self.activations=[]
    
    def forward(self, x):
        x = self.flat(x)
        x = self.layer1(x)
        self.add_activations(x)  # Store first layer activations before ReLU
        x = self.act1(x)
        x = self.layer2(x)
        self.add_activations(x)  # Store second layer activations before ReLU
        x = self.act2(x)
        x = self.layer3(x)
        self.add_activations(x) 
        x = self.softmax(x)
        return x
    
    def l1_activation_loss(self):
        """
        Compute L1 regularization loss for activations
        """
        if len(self.activations) == 0:
            return torch.tensor(0.0)
        
        # L1 loss on activations (before ReLU)
        norms = [torch.norm(act).sum() for act in self.activations]
        l1_loss = torch.tensor(norms).sum()

        return self.l1_lambda * l1_loss
    

class MnistModelAutoencoder(nn.Module):
    def __init__(self, input_size, num_hidden):
        super(MnistModel, self).__init__()
        w = h = int(input_size ** 0.5)
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
        self.reshape = torch.reshape(input_size,input_size)

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
        x = self.reshape(x)
        return x


class MnistDataset(Dataset):
    
    def __init__(self, train=True, device=None, indices=None):
        self.train=train
        self.indices=indices
        self.ds = datasets.MNIST('./data', train=train, download=True)
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
        return MnistDataset(self.train, self.device, indices=indices)

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
    
