

from dataprovider.dataprovider import FashionMnistDataProvider
from dataprovider.example_models import FashionMnistModel


from modularnet.modularspacevisualizer import ModularSpaceVisualizer
from modularnet.modularstep import step_cluster, step_data, step_purge, step_space, step_train, step_validation
from modularnet.modularspace import ModularSpace
from modularnet.modularstats import ModularStats
from utils.utils import seed_everything, session_path


def main():
    name = 'fashion_mnist_mlp'
    seed = 42
    
    bs = 32
    train=False
    space=False
    data=False
    cluster=True
    purge=True

    
    base_path = session_path(name)

    if seed is not None: seed_everything(seed)
    
    space_size = 6
    mnist_dp = FashionMnistDataProvider(autoencoder=False, device='cuda', batch_size=bs)
    mnist_model = FashionMnistModel(mnist_dp.num_features, mnist_dp.num_hidden, mnist_dp.num_classes, device='cuda') #Type: nn.Module
    mnist_space = ModularSpace(name, mnist_model, space_size, device='cuda', useActivations=True, useLayers=True, base_path=base_path)
    mnist_stats = ModularStats(mnist_space, base_path = base_path)
    vis = ModularSpaceVisualizer(mnist_space, base_path = base_path)


    mnist_space.rescale_factor = 0.9
    mnist_space.observing = True
    mnist_space.modularizing = True

    mnist_stats.metric_count(False)
    mnist_stats.metric_covariance(False)

    if train:
        step_train(mnist_dp, mnist_model, mnist_space, epochs=10)

        #mnist_stats.metric_count(True)
        #mnist_stats.metric_covariance(True)
        #step_train(mnist_dp, mnist_space, epochs=1)
        #mnist_stats.show_metric_count()
        #mnist_stats.cluster_space_dbscan()
        #mnist_stats.show_metric_covariance()

        #mnist_stats.metric_count(False)
        #mnist_stats.metric_covariance(False)

    #mnist_stats.stats()


    if space: 
        step_space(mnist_dp, mnist_model, mnist_space, epochs=1)
        vis.play()
        
    
    if data:  
        ys, y_hats, avgs, stds = step_data(mnist_dp, mnist_model, mnist_space)
        vis.show_data_per_class(ys, y_hats, avgs, stds)
    
    if cluster:
        step_cluster(mnist_dp, mnist_model, mnist_space)
        vis.show_cluster_class_attribution()

    
    
    if purge:
        step_purge(mnist_dp, mnist_model, mnist_space)

if __name__ == '__main__':
    main()
    