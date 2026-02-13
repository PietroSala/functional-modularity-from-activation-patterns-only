
import os
import matplotlib
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import math
from utils.utils import session_path
from itertools import count
matplotlib.use('TkAgg')

base_path = session_path('main_demo')

def main():
    simple()


def simple():
    space_size = 2

    activation_pattern = 'always' # 'always', 'uniform', 'normal', 'skewed'
    group_pattern = 'tree' # 'window', 'graph', 'tree'
    
    steps = 500
    #lr_steps = [20,40,80,100]
    lr_steps = [300]# [20,40,80,100]
    #lr_steps = [50,100,150]

    # window/graph
    num_units = 100
    num_groups = 10 
    overlap_size = 0.2 # Range: [0, 1]

    # graph
    num_shared = 4 # Range: [0, num_groups]
    
    # tree
    tree_depth = 3
    tree_num_branch = 3
    tree_num_units = 5
    tree_leafs_only = True
    

    #
    if (group_pattern == 'window'):
        groups = group_partition_window(num_units, num_groups, overlap_size)
    elif (group_pattern == 'graph'):
        groups = group_partition_graph(num_units, num_groups, overlap_size, num_shared)
    elif (group_pattern == 'tree'):
        groups = group_partition_tree(tree_depth, tree_num_branch, tree_num_units, tree_leafs_only)
        [print(group) for group in groups]
        all_groups = []
        for group in groups: all_groups += group
        num_units = len(set(all_groups))
        num_groups = len(groups)
    

    group_weights = group_activation(num_groups, activation_pattern)
    print('group_weights',activation_pattern,group_weights)
    
    
    print("partitions", len(groups), len(groups[0]))

    positions = torch.rand((num_units, space_size))
    positions.requires_grad = True
    history = []

    #optim = torch.optim.SGD([positions], lr=0.1)
    optim = torch.optim.AdamW([positions], lr=0.2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=lr_steps, gamma=0.5)
    for step in range(steps):
        acts = torch.rand((num_groups))
        for i,group in enumerate(groups):
            if acts[i] > group_weights[i]: continue
            idxs = torch.zeros((num_units), dtype=torch.bool)
            idxs[group] = True
            good = positions[idxs]
            bad = positions[~idxs]
            
            #print("good",len(good))
            #print("bad",len(bad))

            update_positions(optim,good,bad)
        lr_scheduler.step()
        history.append( positions.detach().clone().numpy() )

    path_name = f'{base_path}/history_{group_pattern}_u{num_units}_g{num_groups}_s{num_shared}_o{overlap_size*100}.mp4'
    
    show_history(history, path_name)



def group_activation(num_groups, activation_pattern):
    weights = None
    if activation_pattern == 'always':
        weights = torch.ones((num_groups))
    elif activation_pattern == 'uniform':
        weights = torch.rand((num_groups))
    elif activation_pattern == 'normal':
        weights = torch.randn((num_groups))
        weights = ( weights - weights.min() + 0.1 ) / (weights.max() - weights.min())
    elif activation_pattern == 'skewed':
        weights = torch.rand((num_groups))**2
    
    weights = weights / weights.sum()
    weights *= num_groups
    return weights


def update_positions(optim, good, bad):
    avg_good = torch.mean(good,dim=0)
    avg_bad = torch.mean(bad,dim=0)
    
    # near
    diff_good = good-avg_good
    dist_good = diff_good.pow(2).sum(dim=1).sqrt()
    dist_min = (dist_good).pow(2).sum()
    #dist_min = (dist_good * weight_good).sum()

    
    # far
    diff_bad = bad-avg_good
    dist_bad = diff_bad.pow(2).sum(dim=1).sqrt() 
    #dist_max = ((torch.log(1+dist_bad)+dist_bad)/2).sum()
    dist_max = dist_bad.sum()
    dist_max = (torch.log2(1+dist_bad)).sum()
    

    # near, min radius
    distances = torch.cdist(good, good).pow(2)
    mask = torch.eye(good.shape[0], device=good.device).bool()
    distances = distances.masked_fill(mask, float('inf'))
    # Calculate the loss for pairs that are too close
    overlap_loss = torch.relu(0.1 - distances).sum()


    center_diff = (avg_good.detach() - avg_bad).pow(2).sum()
    center_loss = center_diff * torch.log2(1+center_diff)

    loss = dist_min - (dist_max * 1) - (center_loss*2) #+ (overlap_loss * 0.1) 

    center_diff = (avg_good - avg_bad.detach() ).pow(2).sum()
    center_loss = (torch.log2(1+center_diff))

    loss = dist_min - dist_max  * 10 - center_loss #+ (overlap_loss * 0.5)
    loss.backward()
    
    optim.step()

    optim.zero_grad()

def show_history(history, path='history.mp4'):
    history = np.array(history)
    min_x = np.min(history[...,0])
    max_x = np.max(history[...,0])
    min_y = np.min(history[...,1])
    max_y = np.max(history[...,1])
    # initial data
    plt.ion()
    # creating the first plot and frame
    fig, ax = plt.subplots()    
    graph = ax.scatter([],[])
    
    
    
    # updates the data and graph
    def update(frame):
        i, frame_data = frame
        ax.clear()  # clearing the axes
        ax.set_title(f'Frame {i}')  # updating the title
        ax.set_xlim(xmin=min_x, xmax=max_x)
        ax.set_ylim(ymin=min_y, ymax=max_y)
        x = frame_data[:,0]
        y = frame_data[:,1]
        ax.scatter(x,y, s = 100, c = 'b', alpha = 0.1)  # creating new scatter chart with updated data
        fig.canvas.draw()
    
    anim = FuncAnimation(fig, update, frames = enumerate(history), save_count= len(history), interval=0.01)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("saving video:", path)
    anim.save(path, fps=30)
    plt.show(block=True)




def group_partition_window(num_units, num_groups, overlapping, clip_groups=True):
    positions_idxs = np.array(list(range(num_units)))
    
    group_size = math.ceil(num_units / num_groups)
    overlap_size = math.ceil(group_size * overlapping)
    stride_size = group_size - math.ceil(overlap_size / 4 )
    #print("split",windows_size, overlap_size, stride_size )
    group_size += overlap_size
    
    groups = view_as_windows( positions_idxs, int(group_size), int(stride_size))
    if clip_groups:
        groups = groups[:num_groups]
    groups = groups.tolist()
    print("groups")
    print(groups)
    return groups


def group_partition_graph(num_units, num_groups, overlap_size, overlap_num):
    positions_idxs = np.array(list(range(num_units)))
    
    groups = np.array_split(positions_idxs, num_groups)
    
    group_size = math.ceil(num_units / num_groups)

    ols = math.ceil(group_size * overlap_size) 
    
    shared = np.array(groups)[:,-ols:]

    #print("shared",shared)

    print(groups)
    gs = []
    for i, s in enumerate(shared):
        gs.append([i])
        for k in range(overlap_num):
            g = (i+k+1) % num_groups
            gs[i].append(g)
            ng = np.concatenate([groups[g], s])
            groups[g] = ng
    print("gs", gs)
    print("groups")
    print(groups)
    return groups

def group_partition_tree(depth, num_branch, num_units, leafs_only=True):
    
    root = make_tree(depth, num_branch, num_units)
    [print(node.depth,node.data) for node in root.all_nodes]
    groups = [node.data for node in root.all_nodes if (node.depth == (depth-1) or not leafs_only) ]
    return groups

class Node:

    def __init__(self, num_units, parent=None) -> None:
        self.children = []
        self.parent = parent
        self.all_nodes = [] if parent is None else parent.all_nodes
        self.counter = count(0) if parent is None else parent.counter
        self.depth = 0 if parent is None else parent.depth + 1
        self.data = [next(self.counter) for i in range(num_units)]
        if parent is not None: self.data = parent.data + self.data

        self.all_nodes.append(self)



def make_tree(depth, num_branch, num_units, parent=None):
    if depth == 0: return None

    node = Node(num_units, parent)


    #if depth == 1: return node

    for i in range(num_branch):
        child = make_tree(depth-1, num_branch, num_units, node)
        node.children.append(child)

    return node


if __name__ == '__main__': main()