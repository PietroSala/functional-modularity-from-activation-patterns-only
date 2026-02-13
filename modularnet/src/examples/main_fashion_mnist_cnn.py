

from dataprovider.example_models import FashionMnistCNNModel
from dataprovider.mnist_fashion import FashionMnistDataProvider


from modularnet.modularspacevisualizer import ModularSpaceVisualizer
from modularnet.modularstep import step_data, step_space, step_train
from modularnet.modularspace import ModularSpace
from modularnet.modularstats import ModularStats
from utils.utils import seed_everything

""" modularize
  warnings.warn(warn_msg)
[   0/10 | Train L:0.01404 A:0.84 | Val L:0.01309 A:0.85 ]
1it [00:12, 12.97s/it][   1/10 | Train L:0.00955 A:0.89 | Val L:0.00945 A:0.89 ]
2it [00:25, 12.43s/it][   2/10 | Train L:0.00873 A:0.90 | Val L:0.00903 A:0.90 ]
3it [00:37, 12.46s/it][   3/10 | Train L:0.00808 A:0.91 | Val L:0.00870 A:0.90 ]
4it [00:49, 12.29s/it][   4/10 | Train L:0.00767 A:0.91 | Val L:0.00901 A:0.90 ]
5it [01:01, 12.27s/it][   5/10 | Train L:0.00731 A:0.92 | Val L:0.00941 A:0.90 ]
6it [01:13, 12.20s/it][   6/10 | Train L:0.00684 A:0.92 | Val L:0.00926 A:0.90 ]
7it [01:25, 12.14s/it][   7/10 | Train L:0.00664 A:0.92 | Val L:0.00990 A:0.89 ]
8it [01:38, 12.27s/it][   8/10 | Train L:0.00639 A:0.93 | Val L:0.00931 A:0.90 ]
9it [01:50, 12.19s/it][   9/10 | Train L:0.00608 A:0.93 | Val L:0.00908 A:0.91 ]
10it [02:02, 12.28s/it]
0it [00:00, ?it/s][   0/1 | Train L:0.00606 A:0.93 | Val L:0.00957 A:0.90 ]
1it [01:25, 85.32s/it]                    
"""


""" vanilla 
[   0/10 | Train L:0.01404 A:0.84 | Val L:0.01309 A:0.85 ]
1it [00:11, 11.11s/it][   1/10 | Train L:0.01002 A:0.89 | Val L:0.00973 A:0.89 ]
2it [00:21, 10.60s/it][   2/10 | Train L:0.00882 A:0.90 | Val L:0.00901 A:0.90 ]
3it [00:31, 10.51s/it][   3/10 | Train L:0.00809 A:0.91 | Val L:0.00863 A:0.90 ]
4it [00:42, 10.68s/it][   4/10 | Train L:0.00747 A:0.91 | Val L:0.00883 A:0.90 ]
5it [00:53, 10.70s/it][   5/10 | Train L:0.00704 A:0.92 | Val L:0.00898 A:0.90 ]
6it [01:03, 10.60s/it][   6/10 | Train L:0.00662 A:0.92 | Val L:0.00898 A:0.91 ]
7it [01:13, 10.39s/it][   7/10 | Train L:0.00632 A:0.93 | Val L:0.00990 A:0.89 ]
8it [01:24, 10.43s/it][   8/10 | Train L:0.00595 A:0.93 | Val L:0.00945 A:0.90 ]
9it [01:34, 10.27s/it][   9/10 | Train L:0.00577 A:0.93 | Val L:0.00946 A:0.90 ]
10it [01:44, 10.49s/it]
0it [00:00, ?it/s][   0/1 | Train L:0.00557 A:0.93 | Val L:0.00909 A:0.91 ]
1it [01:22, 82.33s/it]
"""

def main():
    seed = 17
    train=False
    space=False
    data=True
    
    if seed is not None: seed_everything(seed)
    
    space_size = 4
    mnist_dp = FashionMnistDataProvider(batch_size=32)
    mnist_model = FashionMnistCNNModel(num_kernels=64, num_classes=mnist_dp.num_classes) #Type: nn.Module
    mnist_space = ModularSpace('fashion_mnist_cnn', mnist_model, space_size, useActivations=False, useLayers=True)
    mnist_stats = ModularStats(mnist_space)
    vis = ModularSpaceVisualizer(mnist_space)

    mnist_space.rescale_factor = 0.7
    mnist_space.observing = True
    mnist_space.modularizing = True
    
    if train:
        step_train(mnist_dp, mnist_model, mnist_space, epochs=10)


        mnist_stats.metric_count(True)
        mnist_stats.metric_covariance(True)
        step_train(mnist_dp, mnist_model, mnist_space, epochs=1)
        mnist_stats.show_metric_count()
        mnist_stats.show_metric_covariance()
        
        vis.play()
    #minst_stats.stats()
    #minst_stats.cluster_space_dbscan()
    
    #
    #mnist_space.clear_history()
    if space: 
        step_space(mnist_dp, mnist_model, mnist_space, epochs=1)
        vis.play()
    
    if data: 
        step_data(mnist_dp, mnist_model, mnist_space)
        

    

if __name__ == '__main__':
    main()
    