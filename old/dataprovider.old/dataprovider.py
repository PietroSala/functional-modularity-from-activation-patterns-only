import os
import torch
from torch.utils.data import Dataset, DataLoader

class BaseDataProvider():
    model: torch.nn.Module

    def __init__(self, 
                 batch_size=32,
                 shuffle=True,
                 split=[0.7, 0.2, 0.1],
                 device = None
                 ):
        self.name = '-base-'
        self.task_type = 'classification' # 'classification' | 'regression' | 'autoencoder'
        self.num_features = None
        self.num_classes = None
        self.criterion = None

        self.dataset_split = split
        self.dataset = None
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = None
        self.val = None
        self.test = None
        
        self.model = None 
        self.autoencoder = False

        self.device = device
        if self.device is None: 
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        

    def save(self, path=None):
        if path is None: path=f'checkpoint/{self.name}_model.pth'
        if os.path.exists(path): os.remove(path)
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        if path is None: path=f'checkpoint/{self.name}_model.pth'
        if not os.path.exists(path): return
        self.model.load_state_dict(torch.load(path, weights_only=True))
