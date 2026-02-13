from dataprovider.example_models import IrisModel
import torch
from torch import nn

from dataprovider.iris import IrisDataProvider
from modularnet.modularspacevisualizer import ModularSpaceVisualizer
from modularnet.modularspace import ModularSpace

num_hidden = 20
space_size = 2

#TODO: convert to modulatstep imports

def main():
    #seed_everything(42)
    #iris_train()
    iris_infer()
    iris_eval()
    #train_mnist()
    #input("Press Enter to close...")
    

def iris_eval():
    bs = 32

    iris = IrisDataProvider(batch_size=bs,  shuffle=False)
    iris_model = IrisModel(iris.num_features,  iris.num_classes)
    iris_model.load()
    
    space = ModularSpace(iris.name, iris_model, space_size)
    space.load_space()
    vis = ModularSpaceVisualizer(space)

    iris_model.eval()


    space.add_history()
    
    ys = []
    y_hats = []
    avgs = []
    stds = []
    for x, y in iris.train:
        y_hat_logits=iris_model(x)
        y_hat = y_hat_logits.argmax(dim=1)

        actives, inactives, weights = space.last_actives()
        avg, std = space.rate(actives, weights)
        
        ys.extend(y.tolist())
        y_hats.extend(y_hat.tolist())
        avgs.extend(avg)
        stds.extend(std)
    
    vis.show_data(ys, y_hats, avgs, stds)


def iris_infer():
    epochs = 50
    bs = 3

    iris = IrisDataProvider(batch_size=bs, num_hidden=num_hidden)
    iris.load()
    
    criterion = nn.CrossEntropyLoss()
    
    space = ModularSpace(iris.name, iris.model, space_size, device='cpu')
    vis = ModularSpaceVisualizer(space)

    iris.model.eval()

     
    for epoch in range(epochs):
        epoch_loss = 0
        train_acc = 0
        num = 0
        for x, y in iris.train:
            y_hat = iris.model(x)
            loss = criterion(y_hat, y)
            #print(y, y_hat.argmax(dim=1))
            actives, inactives, weights = space.last_actives()
            space.step(actives, inactives, weights)
            
            with torch.no_grad():
                epoch_loss += loss.detach().item()
                y_hat_cls = torch.argmax(y_hat,dim=1)
                train_acc += torch.sum(y == y_hat_cls)
                num += len(x)
        avg_loss = float(epoch_loss/num)
        avg_acc = float(train_acc/num)
        print('Infer', epoch, avg_loss, avg_acc)
    space.save_space()
    space.stats()
    vis.play()



def iris_train():
    epochs = 100
    lr = 0.1
    bs = 32

    iris = IrisDataProvider(batch_size=bs, num_hidden=num_hidden)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(iris.model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0
        train_acc = 0
        num = 0
        for x, y in iris.train:
            optim.zero_grad()
            y_hat = iris.model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optim.step()
            
            with torch.no_grad():
                epoch_loss += loss.detach().item()
                y_hat_cls = torch.argmax(y_hat,dim=1)
                train_acc += torch.sum(y == y_hat_cls)
                num += len(x)
        avg_loss = float(epoch_loss/num)
        avg_acc = float(train_acc/num)
        print('Train', epoch, avg_loss, avg_acc)
    
    iris.save()

    #vis.play()
    

if __name__ == '__main__':
    main()
    