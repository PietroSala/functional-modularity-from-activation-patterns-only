import time
import numpy as np
import torch
import os


debug = False

last_session_path = None

def session_path(name, uid=None, createDirectory = True, force=False):   
    global last_session_path 
    if last_session_path is not None and not force: return last_session_path
    
    basedir = 'output/'
    uid = str(uid) if uid is not None else str(time.time()).replace('.','_')
    path = basedir + name + "_" + uid + "/"
    last_session_path = path
    if createDirectory: os.makedirs(path, exist_ok=True)
    return path


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def check_tensor(tensor, name="tensor", debug = debug):
    if not debug: return
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"NaN detected in {name}")
        info_tensor(tensor, name)
        raise Exception("NaN detected")
    
def info_tensor(tensor, name="tensor", threshold=1e-10, debug = debug):
    if not debug: return
    print(f"-----")
    print(f"Tensor: {name}")
    print(f"Shape: {tensor.shape}")
    print(f"Min: {tensor.min()}")
    print(f"Max: {tensor.max()}")
    print(f"Mean: {tensor.mean()}")
    if torch.isnan(tensor).any().item(): print("NaNs!")
    if torch.isinf(tensor).any().item(): print("Infs!")
    near_zero_count = (torch.abs(tensor) < threshold).sum()
    if near_zero_count>0: print(f"Zeros: {near_zero_count}")
    print(f"=====")

def module_name(module):
    return module.__class__.__module__

def class_name(module):
    return module.__class__.__name__

def is_activation(module):
    return module_name(module) == 'torch.nn.modules.activation'

def is_layer(module):
    name = module_name(module).lower()
    layers = [
        'torch.nn.modules.linear',
        'torch.nn.modules.conv',
        'torch.nn.modules.rnn',
        'torch.nn.modules.transformer'
    ]
    return name in layers

def is_conv(module):
    return module_name(module) == 'torch.nn.modules.conv'

def truth_table_from_cm(confusion_matrix:np.ndarray):
    truth_table = []
    for i in range(len(confusion_matrix)):
        tp = confusion_matrix[i, i] # True Positives: diagonal element
        fp = np.sum(confusion_matrix[:, i]) - tp # False Positives: sum of column i, excluding diagonal
        fn = np.sum(confusion_matrix[i, :]) - tp # False Negatives: sum of row i, excluding diagonal  
        tn = np.sum(confusion_matrix) - tp - fp - fn # True Negatives: total - tp - fp - fn
        truth_table.append(np.array((tp, fp, fn, tn)))
    return  np.array(truth_table)

def f1s_from_truth_table(tt:np.ndarray):
    f1s = []
    for tp, fp, fn, tn in tt:
        f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) > 0 else 0
        f1s.append(f1)
    return np.array(f1s)



def set_best(bests, key, val, min=True):
    if key not in bests:
        bests[key] = val
        return 1, None
    
    if (min and val < bests[key]) or (not min and val > bests[key]):
        diff = 1 if key not in bests else abs((bests[key] - val) / bests[key])
        old_val = bests[key]
        bests[key] = val
        return diff, old_val

    return -1, bests[key]

