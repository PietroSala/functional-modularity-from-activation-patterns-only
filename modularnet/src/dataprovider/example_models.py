
from modularnet.metamodel.metamodel import BaseModel
from torch import nn
import torch


class FashionMnistModel(BaseModel):
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

class FashionMnistCNNModel(BaseModel):
    
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

class FashionMnistAutoencoderModel(BaseModel):
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


class MnistModel(BaseModel):
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

class MnistModelSpase(BaseModel):
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
    

class MnistModelAutoencoder(BaseModel):
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


class IrisModel(BaseModel):
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
