import matplotlib
import torch
from typing import *

import os


from modularnet.modularspace import ModularSpace, is_layer


class StrangeSpaceEditor:
    def __init__(self,  space: ModularSpace):
        matplotlib.use('TkAgg')
        self.space = space
        self.operations=[]
    
    

    def saveSpace(self, path, positions=None ):
        if os.path.exists(path): os.remove(path)
        if positions is None: positions = self.space.positions 
        torch.save(positions, path)

    def loadSpace(self, path):
        self.positions = torch.load(path)

    def saveModel(self, path, model=None):
        if os.path.exists(path): os.remove(path)
        if model is None: model = self.space.module 
        torch.save(model.state_dict(), path)

    def loadModel(self, path):
        self.space.module.load_state_dict(torch.load(path))
    

    def splitOutskirt(self):
        module = self.space.module