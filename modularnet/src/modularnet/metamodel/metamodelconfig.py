from hashlib import md5
import json
import os
from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig
import yaml


class MetaModelTemplate:
    CONFIG_PATH='experiments/config/'
    def __init__(self, name):
        self.name = name
        self.params = {}
    
    def get_params(self):
        return list(self.params.values())

    def add_range(self, name: str, bounds, parameter_type, log_scale = False):
        #param = RangeParameterConfig(name, bounds, parameter_type, step_size, scaling)
        param = {"name": name, "value_type": parameter_type, "type": "range", "bounds": bounds, "log_scale": log_scale}
        self.params[name] = param
    
    def add_choice(self, name: str, values: list[float] | list[int] | list[str] | list[bool], parameter_type, is_ordered: bool | None = None, dependent_parameters: dict = None):
        #param = ChoiceParameterConfig(name, values, parameter_type, is_ordered, dependent_parameters)
        param = {"name": name, "value_type": parameter_type, "type": "choice", "values": values, "is_ordered": is_ordered, "dependents": dependent_parameters}
        self.params[name] = param

    def add_fixed(self, name: str, value: list[float] | list[int] | list[str] | list[bool], parameter_type ):
        #param = ChoiceParameterConfig(name, values, parameter_type, is_ordered, dependent_parameters)
        param = {"name": name, "value_type": parameter_type, "type": "fixed", "value": value}
        self.params[name] = param

    
    def signature(self):
        content = json.dumps(self.params, sort_keys=True, default=str)
        content = content.encode()
        sign = md5(content).hexdigest()
        return sign

    @classmethod
    def from_dict(cls, data: dict) -> 'MetaModelTemplate':
        tpl = cls(name=data.get('name', 'default_template'))
        tpl.params = data.get('params', {})
        return tpl
    
    def to_dict(self):
        return {'name': self.name, 'params': self.params}

    def save(self, filepath=None) -> None:
        """
        Save the options to a YAML file.
        """
        if filepath is None:
            filepath = f"{MetaModelTemplate.CONFIG_PATH}/template_{self.name}.yaml"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            
            yaml.dump(self.to_dict(), f)
    
    @classmethod
    def load(cls, filepath) -> 'MetaModelTemplate':
        """
        Load options from a YAML file and return a MetaModelTemplate instance.
        """
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class MetaModelConfig:
    """
    A class to hold options for the MetaModel.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, options):
        self.options = options

        self.exp_metric_modularity = options.get('exp_metric_modularity', True)
        self.exp_modularizer = options.get('exp_modularizer', True)

        
        self.task = options.get('task', "classification")  # Default task is classificationù
        self.criterion_type = options.get('criterion_type', "mse")  # Default task is classification
        self.model_type = options.get('model_type', "fc")  # Default model type is fully connected (fc)
        self.train_epochs = options.get('train_epochs', 10)  # Default number of training epochs
        self.space_epochs = options.get('space_epochs', 1)  # Default number of space epochs ( optimized for unchaning model weights)

        self.dataset_name = options.get('dataset_name', 'fashion_mnist')  # Default input data shape for MNIST
        self.modular_space_name = options.get('modular_space_name', 'fashion_mnist_cnn')  # Default modular space name
        self.modular_space_size = options.get('modular_space_size', 6)  # Default modular space size
        
        self.input_data_shape = options.get('input_data_shape', 784) 
        self.input_hidden_dim = options.get('input_hidden_dim', 32) 

        self.input_num_layers = options.get('input_num_layers', 2)
        self.input_reduce_every = options.get('input_reduce_every', 0)  # Default input size reduced every N layer. <1 no reduction
        self.input_dropout_every = options.get('input_dropout_every', 0) # 0 first only, -1 none, N>0 every N layer)
        self.input_batch_norm = options.get('input_batch_norm', True)
        self.input_residual = options.get('input_residual', True)


        self.output_data_shape = options.get('output_data_shape', 10) # -1: auto
        self.output_hidden_dim = options.get('output_hidden_dim', 32) # -1: auto
        self.output_num_layers = options.get('output_num_layers', 2) # -1: auto
        self.output_reduce_every = options.get('output_reduce_every', -1)  # Default input size reduced every N layer. <1 no reduction
        self.output_dropout_every = options.get('output_dropout_every', 0) # 0 first only, -1 none, N>0 every N layer)
        self.output_batch_norm = options.get('output_batch_norm', True)
        self.output_residual = options.get('output_residual', True)


        
        self.activation = options.get('activation', 'relu')
        self.dropout = options.get('dropout', 0.5)
        self.learning_rate = options.get('learning_rate', 0.001)
        self.batch_size = options.get('batch_size', 32)
        self.epochs = options.get('epochs', 10)


        self.nas_min_acc = options.get('nas_min_acc', 0.8)
        self.nas_min_mod_score = options.get('nas_min_mod_score', 0.2)
        #self.optimizer = options.get('optimizer', 'adam')
        #self.cnn_kernel_size = options.get('kernel_size', 3)

    def signature(self):
        content = json.dumps(self, sort_keys=True, default=str)
        content = content.encode()
        sign = md5(content).hexdigest()
        return sign

    def save(self, filepath):
        """
        Save the options to a YAML file.
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.options, f)

    @classmethod
    def load(cls, filepath):
        """
        Load options from a YAML file and return a MetaModelOptions instance.
        """
        with open(filepath, 'r') as f:
            options = yaml.safe_load(f)
        return cls(**options)
    




class TemplateFashionMnist(MetaModelTemplate):
    VARIANTS = [ 
        'mnist_test_a', 'mnist_test_b', 'mnist_test_c', 'mnist_test_d',
        'fashion_mnist_a', 'fashion_mnist_b', 'fashion_mnist_c', 'fashion_mnist_d',
        'mnist_a', 'mnist_b', 'mnist_c', 'mnist_d',
    ]


    def __init__(self, name: str):
        super().__init__(name)   

        self.add_fixed('exp_metric_modularity', True, 'bool')
        self.add_fixed('exp_modularizer', True, 'bool')
                
        self.add_fixed('task', "classification", "str")  # Default task is classification
        self.add_fixed('criterion_type', "cross_entropy", "str")  # Default task is classification
        self.add_fixed('model_type', "fc", "str")  # Default model type is fully connected (fc])
        self.add_range('train_epochs', [1,10], "int")  # Default number of training epochs
        self.add_fixed('space_epochs', 1, "int")  # Default number of space epochs ( optimized for unchaning model weights)

        self.add_fixed('dataset_name', 'fashion_mnist', "str")  # Default input data shape for MNIST
        self.add_fixed('modular_space_name', 'fashion_mnist', "str")  # Default modular space name
        self.add_range('modular_space_size', [3,8], "int")  # Default modular space size
         
        self.add_fixed('input_data_shape', 784, "int") 
        self.add_range('input_hidden_dim', [16,256], "int") 

        self.add_range('input_num_layers', [2,10], "int")
        self.add_range('input_reduce_every', [2,6], "int")  # Default input size reduced every N layer. <1 no reduction
        self.add_choice('input_dropout_every', [-1, 0], "int") # 0 first only, -1 none, N>0 every N layer])
        self.add_choice('input_batch_norm', [True,False], "bool")
        self.add_choice('input_residual', [True,False], "bool")


        self.add_fixed('output_data_shape', 10, "int") # -1: auto
        self.add_range('output_hidden_dim', [32,128], "int") # -1: auto

        self.add_range('output_num_layers', [2,4], "int") # -1: auto
        self.add_range('output_reduce_every', [0,2], "int")  # Default input size reduced every N layer. <1 no reduction
        self.add_choice('output_dropout_every', [-1,0], "int") # 0 first only, -1 none, N>0 every N layer])
        self.add_choice('output_batch_norm', [True,False], "bool")
        self.add_choice('output_residual', [True,False], "bool")

        self.add_choice('activation', ['relu'], "str")
        self.add_range('dropout', [0.0,0.5], "float")
        self.add_range('learning_rate', [0.00001, 0.001], "float", log_scale=True)
        self.add_range('batch_size', [32,64], "int")

        self.add_fixed('nas_min_acc', 0.8, "float")
        self.add_fixed('nas_min_mod_score', 0.2, "float")
        #self.add_choice('optimizer', ['adam'], "str")
        #self.add_range('kernel_size', [3,9], "int")

    @staticmethod
    def get_experiment_variant(variant:str):
        if variant not in TemplateFashionMnist.VARIANTS:
            raise ValueError(f"Unknown variant: {variant}")
        
        template = TemplateFashionMnist(variant)
            
        if variant.startswith('mnist_'):
            template.add_fixed('dataset_name', 'mnist', 'str')
            template.add_fixed('space_name', 'mnist', 'str')
            template.add_fixed('nas_min_acc', 0.95, "float")

        if variant.endswith('_a'):
            template.add_fixed('exp_metric_modularity', False, 'bool')
            template.add_fixed('exp_modularizer', False, 'bool')
        elif variant.endswith('_b'):
            template.add_fixed('exp_metric_modularity', True, 'bool')
            template.add_fixed('exp_modularizer', False, 'bool')
        elif variant.endswith('_c'):
            template.add_fixed('exp_metric_modularity', False, 'bool')
            template.add_fixed('exp_modularizer', True, 'bool')
        elif variant.endswith('_d'):
            template.add_fixed('exp_metric_modularity', True, 'bool')
            template.add_fixed('exp_modularizer', True, 'bool')
        
        return template


def dump():
    template = TemplateFashionMnist('fashion_mnist')
    template.save('experiments/config/template_fashion_mnist.yaml')


if __name__ == "__main__": dump()

