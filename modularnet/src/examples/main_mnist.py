import time

from dataprovider.example_models import MnistModel
from sklearn.utils import shuffle
import torch
from torch import nn
from torch import device
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataprovider.dataprovider import BaseDataProvider
from dataprovider.iris import IrisDataProvider
from dataprovider.mnist import MnistDataProvider

from modularnet.modularspacevisualizer import ModularSpaceVisualizer
from modularnet.modularstats import ModularStats
from modularnet.modularspace import ModularSpace
from modularnet.modularstep import step_train, step_data, step_space
from utils.utils import seed_everything



def main():
    seed = 17
    train=True
    space=True
    data=False
    
    if seed is not None: seed_everything(seed)
    
    space_size = 2
    mnist_dp = MnistDataProvider()
    mnist_model = MnistModel(mnist_dp.num_features, mnist_dp.num_hidden, mnist_dp.num_classes) #Type: nn.Module
    mnist_space = ModularSpace('mnist_mlp', mnist_model, space_size, useActivations=False, useLayers=True)
    mnist_stats = ModularStats(mnist_space)

    mnist_space.rescale_factor = 0.7
    mnist_space.observing = True
    mnist_space.modularizing = True

    if train: step_train(mnist_dp, mnist_model, mnist_space, epochs=9)

    
    mnist_stats.metric_count(True)
    mnist_stats.metric_covariance(True)
    if train: step_train(mnist_dp, mnist_model, mnist_space, epochs=1)
    mnist_stats.show_metric_count()
    mnist_stats.cluster_space_dbscan()
    mnist_stats.show_metric_covariance()
    #minst_stats.stats()

    if space: step_space(mnist_dp, mnist_model, mnist_space)
    if data:  step_data(mnist_dp, mnist_model, mnist_space)

    


if __name__ == '__main__':
    main()
    