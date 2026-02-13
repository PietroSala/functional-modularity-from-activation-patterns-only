
import os
import time

from modularnet.metamodel.metamodelconfig import MetaModelConfig
import torch
from torchvision import transforms

import torch.nn as nn
from collections import OrderedDict

from torchsummary import summary


class BaseModel(nn.Module):
    def __init__(self, name:str=None, device=None):
        super().__init__()
        self.name = self.__class__.__name__ if name is None else name
        self.timestamp = time.time()
        self.basepath = None
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def model_path(self,path=None):
        if path is None and self.basepath is None:
            path = f'./checkpoint/{self.name}.pth'
        elif path is None:
            path = os.path.join(self.basepath,f'{self.name}.pth')
        return path

    def save(self, path=None):
        path = self.model_path(path)
        if os.path.exists(path): os.remove(path)
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = self.model_path(path)
        if not os.path.exists(path): return
        self.load_state_dict(torch.load(path, weights_only=True))


    

class MetaModel(BaseModel):
    """
    A class representing a meta module that can contain multiple sub-modules.
    This class is used to manage and interact with a collection of modules.
    """

    TASK_LIST = ['classification', 'regression','autoencoder']
    LAYER_TYPES = {'fc': nn.Linear, 'cnn': nn.Conv2d, 'decnn': nn.ConvTranspose2d, 'rnn': nn.RNN, 'transformer': nn.TransformerEncoder}
    # TODO: implementre support for 'rnn', 'transformer']
        
    def __init__(self, name, config:MetaModelConfig, device=None):
        super().__init__()
        self.name = name
        self.config = config
        self.criterion = None

        self.nn_input = None
        self.nn_latent = None
        self.nn_output = None

        self.build_model()
        self.to_gpu()

    def forward(self, x):
        x = self.nn_input(x)
        x = self.nn_latent(x)
        x = self.nn_output(x)
        return x

    def to_gpu(self, num=None):
        has_cuda = torch.cuda.is_available()
        if not has_cuda: 
            print("WARNING: No GPU available, CPU fallback")
            self.device = torch.device("cpu")
            self.to("cpu")
        else:
            device_name = "cuda" if num is None else f"cuda:{num}"
            self.device = torch.device(device_name)
            self.to(device_name)

    def print(self):
        summary(self, input_size=(self.config.input_data_shape,), device=str(self.device))
        print(f"Model size: {self.get_model_size():.2f} MB")

    def to_cpu(self):
        self.device = torch.device("cpu")
        self.to("cpu")

    def loss(self, x, y, y_hat):
        if self.criterion is None:
            raise ValueError("Criterion not defined. Please set a criterion before calling loss.")
        return self.criterion.loss(x, y, y_hat)
        
    def build_model(self):
        task = self.config.task
        model_type = self.config.model_type
        self.criterion = MetaCriterion(self.config)

        if self.config.task not in MetaModel.TASK_LIST:
            raise ValueError(f"Task must be one of {MetaModel.TASK_LIST}, got {task}")
        if self.config.model_type not in MetaModel.LAYER_TYPES:
            raise ValueError(f"Model type must be one of {MetaModel.LAYER_TYPES}, got {model_type}")
        
        self.build_model_input()
        self.build_model_latent()
        self.build_model_output()


    ## Build NN INPUT
    def build_model_input(self):
        self.nn_input = nn.Sequential()
        for layer_num in range(self.config.input_num_layers):
            name = f'input_{self.config.model_type}_{layer_num}'
            block = self.build_model_input_block(layer_num)
            self.nn_input.add_module(name, block)
    

    ## Build NN LATENT
    def build_model_latent(self):
        self.nn_latent = nn.Identity()  # Placeholder for latent layer, can be extended later if needed
    
    
    ## Build NN OUTPUT
    def build_model_output(self):
        self.nn_output = nn.Sequential()  # Placeholder for output layer, can be extended later if needed 

        num_layers = self.config.output_num_layers if self.config.task != 'autoencoder' else self.config.input_num_layers

        for layer_num in range(num_layers):
            layer_name = f'output_{self.config.model_type}_{layer_num}'
            if self.config.task == 'autoencoder':
                block = self.build_model_output_block_autoencoder(layer_num)
            else:
                block = self.build_model_output_block(layer_num)
            self.nn_output.add_module(layer_name, block)
        
## General block parts
    def block_activation(self, name):
        # activation   
        if name == 'relu': return  nn.ReLU()
        if name == 'sigmoid': return nn.Sigmoid()
        if name == 'tanh': return  nn.Tanh()
        if name == 'leaky_relu': return nn.LeakyReLU()
        
        raise ValueError(f"Unsupported activation function: {name}")
    
## General block parts
    def block_layer(self, model_type, input_dim, output_dim= None):
        if output_dim is None: output_dim = input_dim 
        # layer
        if model_type == 'fc':
            return nn.Linear(input_dim, output_dim)
        if model_type == 'cnn':
            return nn.Conv2d(input_dim, output_dim, kernel_size=self.config.cnn_kernel_size, stride=1, padding=1)
        if model_type == 'decnn':
            return nn.ConvTranspose2d(input_dim, output_dim, kernel_size=self.config.cnn_kernel_size, stride=1, padding=1)
        #if model_type == 'rnn':
        #     return nn.RNN(input_dim, output_dim, num_layers=self.options.num_layers, batch_first=True)
        # if model_type == 'transformer':
        #     return nn.TransformerEncoderLayer(d_model=self.options.input_hidden_dim, nhead=8, dim_feedforward=self.options.hidden_dim)
        
        raise ValueError(f"Unsupported model type: {model_type}")
    
    def block_batch_norm(self, model_type, input_dim):
        # batch normalization
        if model_type == 'fc':
            return nn.BatchNorm1d(input_dim)
        elif model_type == 'cnn':
            if self.config.task == 'autoencoder':
                return nn.GroupNorm(input_dim, input_dim)
            else:    
                return nn.BatchNorm2d(input_dim) 
        # elif model_type == 'rnn':
        #     return nn.BatchNorm1d(output_dim)  #
        # elif model_type == 'transformer':
        #     return nn.LayerNorm(output_dim)  # LayerNorm
        else:
            raise ValueError(f"Unsupported model type for batch normalization: {model_type}")

## Build INPUT block
    def build_model_input_block(self, layer_num):
        
        reduce_every = self.config.input_reduce_every
        dropout_every = self.config.input_dropout_every
        batch_norm = self.config.input_batch_norm
        input_residual = self.config.input_residual

        reduce_dim = reduce_every > 1 and layer_num % reduce_every == 0
        is_first_layer = layer_num == 0
        is_last_layer = layer_num == self.config.input_num_layers - 1


        # handle input player 
        if is_first_layer: 
            input_dim = self.config.input_data_shape
            output_dim = self.config.input_hidden_dim  
        else:
            last_size = self.last_layer_size()
            input_dim = self.config.input_hidden_dim if last_size is None else last_size[1]
            output_dim = input_dim // 2 if reduce_dim else input_dim
       


        input = OrderedDict()
        
        input['layer'] = self.block_layer(self.config.model_type, input_dim, output_dim)
        input['activation'] = self.block_activation(self.config.activation)
        if batch_norm:
            input['batch_norm'] = self.block_batch_norm(self.config.model_type, output_dim)

        # dropout
        first_dropout = dropout_every == 0 and is_first_layer
        should_dropout = dropout_every > 0 and (layer_num + 1) % dropout_every == 0
        if first_dropout or should_dropout:
            input['dropout'] = nn.Dropout(self.config.dropout)

        block = nn.Sequential(input)
        if input_residual and not is_first_layer and input_dim == output_dim:
            
            block = ResNet(block)

        return block
    
## Build OUTPUT block
    def build_model_output_block(self, layer_num):

        reduce_every = self.config.output_reduce_every
        dropout_every = self.config.output_dropout_every
        batch_norm = self.config.output_batch_norm
        output_residual = self.config.output_residual
        
        reduce_dim = reduce_every > 1 and (layer_num + 1) % reduce_every == 0

        is_first_layer = layer_num == 0
        is_last_layer = layer_num == self.config.output_num_layers - 1


        # handle latent -> output 
        last_size = self.last_layer_size()
        if is_first_layer: 
            input_dim = last_size[1]
            output_dim = self.config.input_hidden_dim 
        elif is_last_layer:
            input_dim = last_size[1]
            output_dim = self.config.output_data_shape 
        else:
            input_dim = self.config.output_hidden_dim if last_size is None else last_size[1]
            output_dim = input_dim // 2 if reduce_dim else input_dim



        output = OrderedDict()
        
        model_type = self.config.model_type
        if self.config.task in ['classification', 'regression']:
            model_type = 'fc'  # For classification and regression, we use fully connected layers
                
        output['layer'] = self.block_layer(model_type, input_dim, output_dim)
        




        output['activation'] = self.block_activation(self.config.activation)
        if batch_norm:
            output['batch_norm'] = self.block_batch_norm(self.config.model_type, output_dim)
        
        # dropout
        first_dropout = dropout_every == 0 and layer_num == 0
        should_dropout = dropout_every > 0 and (layer_num + 1) % dropout_every == 0
        if first_dropout or should_dropout:
            output['dropout'] = nn.Dropout(self.config.dropout)

        block = nn.Sequential(output)
        if output_residual and not is_last_layer and input_dim == output_dim:
            block = ResNet(block)
        

        return block

    def build_model_output_block_autoencoder(self, layer_num):

        dropout_every = self.config.output_dropout_every
        batch_norm = self.config.output_batch_norm
        output_residual = self.config.output_residual

        num_layers = self.config.input_num_layers
        is_first_layer = layer_num == 0
        is_last_layer = layer_num == num_layers - 1

        # handle latent -> output 
        last_size = self.last_layer_size()
        # if is_first_layer: 
        #     input_dim = last_size[1]
        #     output_dim = self.options.input_hidden_dim 
        # elif is_last_layer:
        #     input_dim = last_size[1]
        #     output_dim = self.options.output_data_shape 
        # else:
        cur_block = self.nn_input[((num_layers - 1) - layer_num)]
        if isinstance(cur_block, ResNet): cur_block = cur_block.module  # Unwrap ResNet if it is a residual block
        input_dim, output_dim = self.layer_size(cur_block.layer)
        input_dim, output_dim = output_dim, input_dim  # Reverse the dimensions for autoencoder



        output = OrderedDict()
        
        model_type = self.config.model_type
        if model_type == 'cnn':
            model_type = 'decnn'
        
        output['layer'] = self.block_layer(model_type, input_dim, output_dim)
        
        output['activation'] = self.block_activation(self.config.activation)
        if batch_norm:
            output['batch_norm'] = self.block_batch_norm(self.config.model_type, output_dim)
        
        # dropout
        first_dropout = dropout_every == 0 and layer_num == 0
        should_dropout = dropout_every > 0 and (layer_num + 1) % dropout_every == 0
        if first_dropout or should_dropout:
            output['dropout'] = nn.Dropout(self.config.dropout)

        block = nn.Sequential(output)
        if output_residual and not is_last_layer and input_dim == output_dim:
            block = ResNet(block)
        

        return block



    ## Utility 

    def last_layer(self):
        out_layer = self.last_output_layer()
        if out_layer is not None: return out_layer
        in_layer = self.last_input_layer()
        return in_layer
    
    def last_layer_size(self):
        out_size = self.last_output_layer_size()
        if out_size is not None: return out_size
        in_size = self.last_input_layer_size()
        return in_size 

    
    def last_input_layer(self):
        if self.nn_input is None: return None
        last_input = list(self.nn_input)
        if len(last_input) == 0: return None
        last_input = last_input[-1]
        
        if last_input is None: return None
        if isinstance(last_input, ResNet):
            last_input = last_input.module
        return last_input.layer
    
    def last_input_layer_size(self):
        li = self.last_input_layer()
        return self.layer_size(li)
    
    def last_output_layer(self):
        if self.nn_output is None: return None
        last_output = list(self.nn_output)
        if len(last_output) == 0: return None
        last_output = last_output[-1]

        if last_output is None: return None
        if isinstance(last_output, ResNet):
            last_output = last_output.module
        return last_output.layer
    
    def last_output_layer_size(self):
        lo = self.last_output_layer()
        return self.layer_size(lo)
        

    def layer_size(self, layer):
        if layer is None: return None
        if isinstance(layer, ResNet):
            layer = layer.module
        if isinstance(layer, nn.Linear):
            return (layer.in_features, layer.out_features)
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return (layer.in_channels, layer.out_channels)
        # elif isinstance(lo, (nn.RNN, nn.LSTM, nn.GRU)):
        #     return (lo.input_size, lo.hidden_size)
        # elif isinstance(lo, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        #     return (lo.d_model, lo.d_model)
        
        raise ValueError(f"Unsupported layer type: {type(layer)}")
    
    def get_model_size(self):
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / 1024**2  # In MB
    
    def nmum_fop(self):
        """
        Returns the number of floating point operations (FLOPs) in the model.
        This is a rough estimate and may not be accurate for all layer types.
        """
        flops = 0
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                flops += layer.in_features * layer.out_features
            elif isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                flops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1]
            # elif isinstance(layer, (nn.RNN, nn.LSTM, nn.GRU)):
            #     flops += layer.input_size * layer.hidden_size
            # elif isinstance(layer, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            #     flops += layer.d_model * layer.d_model
        return flops
        

# resnet utility module
class ResNet(nn.Module):
    def __init__(self, module: nn.Module| OrderedDict):
        super().__init__()
        if isinstance(module, OrderedDict):
            module = nn.Sequential(module)
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
    


class MetaCriterion():
    """
    A class to hold options for the MetaCriterion.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, options):
        self.custom_losses = {}
        self.task =  options.task  # Default task is regression
        self.criterion_type = options.criterion_type

    def custom_loss(self, name, function):
        self.custom_losses[name] = function
    
    def loss(self, x, y, y_hat):
        if self.task == 'autoencoder': y = x

        name = self.criterion_type
        if name == 'cross_entropy':
            return nn.functional.cross_entropy(y_hat, y)        
        elif name == 'mse':
            return nn.functional.mse_loss(y_hat, y)
        elif name == 'mae':
            return nn.functional.l1_loss(y_hat, y)
        elif name == 'huber':
            return nn.functional.huber_loss(y_hat, y)
        elif name == 'bce_with_logits':
            return nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        elif name == 'bce':
            return nn.functional.binary_cross_entropy(y_hat, y)
        elif name in self.custom_losses:
            return self.custom_losses[name](y_hat, y)
        else:
            raise ValueError(f"Unsupported criterion type: {name}")
        



class MetaTransform():
    """
    A class to hold options for the MetaTransform.
    This class can be extended to include various configuration parameters.
    """
    def __init__(self, transform_list):
        self.transform_list = transform_list
        self.transforms = None
        
        ts = []
        for name in self.transform_list:
            
            if name == 'normalize':
                t = transforms.Normalize(mean=[0.5], std=[0.5])
            elif name == 'flatten':
                t = transforms.Lambda(lambda x: torch.flatten(x, start_dim=0))
            elif name == 'to_tensor':
                t = transforms.ToTensor()
            elif name == 'to_float':
                t = transforms.Lambda(lambda x: x.float())
            elif name == 'to_long':
                t = transforms.Lambda(lambda x: x.long())
            elif name == 'normalize_image':
                t = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            elif name == 'augment_image':
                t = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(360),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                raise ValueError(f"Unsupported transform type: {name}")

            ts.append(t)
        self.transforms = transforms.Compose(ts)
    

        
    def apply(self, x):
        
        return self.transforms(x)