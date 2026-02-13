import matplotlib.pyplot as plt
import numpy as np
import time 
import os
import yaml

import torch
import torchvision.transforms.v2 as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import evaluate  # train,
from kitty_dataset import KittiDataset

yaml.add_representer(np.ndarray, lambda dumper, array: dumper.represent_list(array.tolist()))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float
session_id = str(int(time.time()*1000))
output_dir = './output/'
session_dir = output_dir + session_id + '/'
os.makedirs(session_dir, exist_ok=True)
exp_num = 0

def main():
    
    params = [
        {"name": "lr",                 "value_type": "float",  "type": "range", "bounds": [1e-6, 0.4], "log_scale": True}, #optimizer
        {"name": "lr_max",             "value_type": "float",  "type": "range", "bounds": [1e-6, 0.4], "log_scale": True}, #LR scheduler

        {"name": "batch_size",         "value_type": "int",    "type": "range", "bounds": [4, 128]}, #train loop
        {"name": "num_epoch",          "value_type": "int",    "type": "range", "bounds": [1, 30]}, #train loop
        {"name": "drop_out",           "value_type": "float",  "type": "range", "bounds": [0.0, 0.9]}, # model FC
        {"name": "fc_hidden_num",      "value_type": "int",    "type": "range", "bounds": [0, 10]}, # model FC
        {"name": "fc_hidden_size",     "value_type": "int",    "type": "range", "bounds": [64, 2048]}, # model FC
        
        {"name": "resnet_size",        "value_type": "int",    "type": "choice", "values": [18, 34, 50, 101, 152], "sort_values": True, "is_ordered": True}, # model CNN
    ]

    #{'best_parameters': {'lr': 2.2603352426200233e-05, 'lr_max': 0.004585287589188039, 'batch_size': 4, 'num_epoch': 19, 'drop_out': 0.12963303754006938, 'fc_hidden_num': 3, 'fc_hidden_size': 897, 'resnet_size': 101}, 'means': {'loss': 1.5484636355583556}, 'covariances': {'loss': {'loss': 0.02377678650315952}}}

    best_params, stats, experiment, model = optimize(
        parameters=params,
        evaluation_function=train_evaluate,
        objective_name='loss',
        minimize=True
    )


    means, covariances = stats
    results = {
        'params': best_params,
        'means': means,
        'covariances': covariances,
    }
    
    results_yaml = yaml.dump(results) 
    with open(session_dir + 'params.yml', 'w') as f: f.write(results_yaml)


    trainloader = getDataLoader(best_params)
    model_best = build_model(best_params)
    model_best_trained = train_model(best_params, model_best, trainloader)
    
    torch.save(model_best_trained, session_dir + 'model.pth')
    


def train_evaluate(params):
    trainloader = getDataLoader(params)
    valloader = getDataLoader(params, False)

    model = build_model(params)
    model_trained = train_model(params, model, trainloader)
    loss = evaluate_model(model_trained, valloader)

    return loss

def evaluate_model(model, data_loader):
    model.to(dtype=dtype, device=device)
    model.eval()

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()

    total_loss = 0
    num_samples = 0

    for x_images, y_labels in data_loader:
        x_images = x_images.to(dtype=dtype, device=device)
        y_labels = y_labels.to(dtype=dtype, device=device)

        y_hat = model(x_images)
        loss_class = criterion_class(y_hat[:, 0], y_labels[:, 0])
        loss_bbox = criterion_bbox(y_hat[:, 1:], y_labels[:, 1:])
        loss = loss_class + loss_bbox

        total_loss += loss.item()
        num_samples += len(x_images)

    return total_loss / num_samples

def train_model(params, model, data_loader):
    model.to(dtype=dtype, device=device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_bbox = nn.MSELoss()

    lr = params.get('lr', 0.001)
    lr_max = params.get('lr_max', lr*10)
    num_epochs = params.get("num_epochs", 1)
    

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr
        # momentum=params.get('momentum',0.9)
    )
    num_batches = len(data_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr_max, total_steps=num_epochs*num_batches)

    for _ in range(num_epochs):
        for x_images, y_labels in data_loader:
            x_images = x_images.to(dtype=dtype, device=device)
            y_labels = y_labels.to(dtype=dtype, device=device)

            optimizer.zero_grad()
            y_hat = model(x_images)
            loss_class = criterion_class(y_hat[:, 0], y_labels[:, 0])
            loss_bbox = criterion_bbox(y_hat[:, 1:], y_labels[:, 1:])
            loss = loss_class + loss_bbox
            loss.backward()

            optimizer.step()
            scheduler.step()
    return model


def build_model(params):
    dropout = params.get('dropout', 0.1)
    resnet_size = params.get('resnet_size', 18)
    resnet_size = params.get('resnet_size', 18)
    fc_hidden_num = params.get('fc_hidden_num', 0)  # Hidden layer size; you can optimize this as well
    fc_hidden_size = params.get('fc_hidden_size', 64)  # Hidden layer size; you can optimize this as well

    if resnet_size == 18:  model = resnet18(weights=ResNet18_Weights.DEFAULT)
    if resnet_size == 34:  model = resnet34(weights=ResNet34_Weights.DEFAULT)
    if resnet_size == 50:  model = resnet50(weights=ResNet50_Weights.DEFAULT)
    if resnet_size == 101: model = resnet101(weights=ResNet101_Weights.DEFAULT)
    if resnet_size == 152: model = resnet152(weights=ResNet152_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False  # Freeze feature extractor

    in_features = model.fc.in_features
    
    fc_layers = [
        nn.Dropout(dropout),
        nn.Linear(in_features, fc_hidden_size),    
        nn.ReLU() 
    ]

    for _ in range(fc_hidden_num):
        fc_layers.extend([
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_size, fc_hidden_size),    
            nn.ReLU() 
        ])
    
    fc_layers.extend([
        nn.Dropout(dropout),
        nn.Linear(fc_hidden_size, 5),
        nn.Sigmoid()
    ])
        
    model.fc = nn.Sequential(*fc_layers)
    
    return model  # return untrained model


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def getTransforms(params):
    img_size = (370, 1224)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(mean, std)
    ])
    return transform


def getDataLoader(params, train=True, download=False):
    batch_size = params.get('batch_size', 32)
    transform = getTransforms(params)
    dataset = KittiDataset(root='./datasets', train=train, download=download, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True,
                            collate_fn=collate_batch, )  # , pin_memory_device=device)
    return dataloader


def collate_batch(batch):
    x_images = [data[0] for data in batch]
    y_labels = [data[1][0] for data in batch]

    image_sizes = [torch.as_tensor((img.shape[2], img.shape[1], img.shape[2], img.shape[1])) for img in
                   x_images]  # h,w,h,w
    y_bbox = [torch.as_tensor(label.get('bbox', [0, 0, 0, 0])) for label in y_labels]
    y_class = [torch.as_tensor([1 if label['type'] == 'Car' else 0]) for label in y_labels]

    x_images = torch.stack(x_images)
    y_class = torch.stack(y_class)
    y_bbox = torch.stack(y_bbox) / torch.stack(image_sizes)

    y_labels = torch.cat((y_class, y_bbox), dim=1)

    return x_images, y_labels


if __name__ == '__main__': main()
