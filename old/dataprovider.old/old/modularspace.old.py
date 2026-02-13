import torch
from torch import nn
from torch.nn import functional as F
from typing import *
import einops
import math
import os

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

def is_observer(module):
    if isinstance(module, ModularObserver): return True
    return False

class ModularObserver(nn.Module):
    def __init__(self,  space: 'ModularSpace', input_size:int, observed:nn.Module, depth:int, device:torch.device):
        super(ModularObserver, self).__init__()
        self.top_x = 0.05
        self.quantile_x = 0.90
        self.input_size = input_size
        self.space = space
        self.space_size = self.space.space_size
        self.observed = observed
        self.device = device
        self.positions = torch.rand( (input_size, self.space_size), requires_grad=True, device=self.device)
        self.depth = depth
        self.positions_count = torch.zeros(input_size, requires_grad=False, device=self.device)
        self.data_count = 0

        self.active_positions = None
        self.inactive_positions = None
        self.weight_positions = None
        self.goods = None
        self.bads = None
    
    def value(self):
        return self.depth #math.log(1+self.depth)

    def forward(self, x): 
        if self.space.observing == False: 
            return x

        
        bs = x.shape[0]
        self.data_count += bs

        if class_name( self.observed ) == 'Softmax':
            idxs = torch.zeros_like(x, dtype=torch.bool)
            top = x.argmax(dim=1).unsqueeze(1)
            for i, j in enumerate(top): idxs[i,j] = True

            self.active_positions = [self.positions[idx] for idx in idxs]
            self.inactive_positions = [self.positions[~idx] for idx in idxs]

            self.weight_positions = torch.ones(bs, 1, device=x.device)
            for idx in idxs: self.positions_count[idx] += 1 
                
            #self.active_positions[:, ] = 1.0 #self.value() #True
        else:
            #self.active = x.detach()>0
            #top_k = math.floor(self.input_size * self.top_x)
            ax = x.detach()

            

            #vals, idxs = torch.topk(ax, top_k, dim=1)

            q = torch.quantile(ax, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1)
            idxs = (ax >= q)
            
            vals = [ax[i][idx] for i,idx in enumerate(idxs)]
            vals_rel = [val/val.sum() for val in vals]
            pos_good = [self.positions[idx] for idx in idxs]
            pos_bad = [self.positions[~idx] for idx in idxs]
            for idx in idxs: self.positions_count[idx] += 1 
            
            
            self.active_positions = pos_good
            self.inactive_positions = pos_bad
            self.weight_positions = vals_rel

        #position_batch = self.positions.expand( bs, -1, -1 ) 
        
        #actives[self.active_positions] = True
        #row_indices = torch.arange(len(actives)).unsqueeze(1).expand(-1, self.active_positions.size(1))
        #actives[row_indices, self.active_positions] = True
        
        #self.goods = position_batch[actives] 
        #self.bads = position_batch[~actives]

        return x


class ModularSpace():
    def __init__(self, name:str, module: nn.Module, space_size:int, lr:float=1.0, history:bool=True, noise=True, useLayers=False, useActivations=True, device=None):
        self.name = name
        self.max_dist = 2
        self.min_dist = 0.1
        self.module = module
        self.space_size = space_size
        self.noise = noise 
        self.device = torch.device(device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr_steps = [100,200,300]
        self.observing = True

        if useLayers: self.inject_observer_after_layers(self.module)
        if useActivations: self.inject_observer_after_activations(self.module)
        # Layer * 1 * Hidden * Space 
        self.sos = [module for module in module.modules() if is_observer(module)] #type: List[ModularObserver]
        #print("SOS: ",[(sos.depth,sos.input_size) for sos in self.sos])
        self.positions = [sa.positions for sa in self.sos]
        

        self.optim = torch.optim.AdamW(self.positions, lr=lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.lr_steps, gamma=0.5)

        #self.positions = [pos.to(device=self.device) for pos in self.positions]
        self.history = history
        self.position_history = []
    
    def stats(self):
        positions_count = [sa.positions_count for sa in self.sos]
        data_count = [sa.data_count for sa in self.sos]

        for pos, cnt in zip(positions_count, data_count):
            perc = ((pos/cnt)*100)
            q75 = torch.quantile(perc, 0.75).type(torch.int).tolist()
            q50 = torch.quantile(perc, 0.50).type(torch.int).tolist()
            q25 = torch.quantile(perc, 0.25).type(torch.int).tolist()
            perc = perc.type(torch.int).tolist()
            print(q75,q50,q25)
            print(perc)
        
    
    def save(self, path=None):
        if path is None: path=f'checkpoint/{self.name}_space.pth'
        if os.path.exists(path): os.remove(path)
        torch.save(self.positions, path)

    def load(self, path=None):
        if path is None: path=f'checkpoint/{self.name}_space.pth'
        self.positions = torch.load(path, weights_only=True)
    
    def inject_observer_after_activations(self, parent: nn.Module, out_features:int|None=None, depth=1):
        for name, child in parent.named_children():
            if hasattr(child,'out_features'): out_features = child.out_features
            elif hasattr(parent,'out_features'): out_features = parent.out_features

            if is_activation(child):
                ss = ModularObserver(self, out_features, child, depth, self.device)
                new_sequence = nn.Sequential(
                    child,
                    ss
                )
                setattr(parent, name, new_sequence)
            else:
                self.inject_observer_after_activations(child, out_features, depth+1)
            
            depth +=1
            #if hasattr(child,'out_features'): out_features = child.out_features
    
    def inject_observer_after_layers(self, parent: nn.Module, out_features:int|None=None, depth=1):
        for name, child in parent.named_children():
            if hasattr(child,'out_features'): out_features = child.out_features
            elif hasattr(parent,'out_features'): out_features = parent.out_features
                
            if is_layer(child):
                ss = ModularObserver(self, out_features, child, depth, self.device)
                new_sequence = nn.Sequential(
                    child,
                    ss
                )
                setattr(parent, name, new_sequence)
            else:
                self.inject_observer_after_layers(child, out_features, depth+1)
            depth +=1
            
    def last_actives(self):
        actives = [so.active_positions for so in self.sos]
        inactives = [so.inactive_positions for so in self.sos]
        weights = [so.weight_positions for so in self.sos]
        return actives, inactives, weights

    def add_history(self):
        if (self.history):
            pos = [ pos.tolist() for pos in self.positions]
            self.position_history.append( pos )
    


    def rate(self, actives=None, weights=None):
        # Layer * Batch * Hidden * Space
        if actives is None or weights is None:
            actives, inactive, weights = self.last_actives()
        
        self.add_history()
        
        stats_std = []
        stats_avg = []
        
        bs = len(actives[0])
    
        for i in range(bs):
            goods = []
            good_weights=[]

            for active, weight in zip(actives, weights):
                goods.append(active[i])
                good_weights.append(weight[i])

            good = torch.vstack(goods).detach()
            good_weight = torch.concat(good_weights)
            good_weight = (good_weight / good_weight.sum()).unsqueeze(1)

            #unweighted
            #std, avg = torch.std_mean(good, dim=0)

            #weighted
            #avg = torch.mean(good, dim=0).detach()
            avg = torch.sum(good * good_weight, dim=0).detach()
            std = torch.mean( (avg-good).pow(2), dim=0 ).detach()

            
            
            stats_std.append(std.tolist())
            stats_avg.append(avg.tolist())
        
        return stats_avg, stats_std

   

    def step(self, actives=None, inactives=None, weights=None):
        
        # Layer * Batch * Hidden * Spac5e
        if actives is None or inactives is None or weights is None:
            actives, inactives, weights = self.last_actives()

        bs = len(actives[0])
        for i in range(bs):
            goods = []
            bads = []
            good_weights = []
            
            for active, inactive, weight in zip(actives, inactives, weights):
                goods.append(active[i])
                bads.append(inactive[i])
                good_weights.append(weight[i])
                
            good = torch.vstack(goods)
            bad = torch.vstack(bads)
            good_weight = torch.concat(good_weights)

            self.update_space(good, bad, good_weight)
        self.add_history()


    def update_space(self, good:torch.Tensor, bad:torch.Tensor, good_weight:torch.Tensor):

        good_weight = (good_weight / good_weight.sum()).unsqueeze(1)
        
        #avg_good = torch.mean(good, dim=0)#.detach()
        avg_good = torch.sum(good * good_weight, dim=0).detach()
        avg_bad = torch.mean(bad, dim=0)#.detach()

        # near
        diff_good = good-avg_good
        dist_good = diff_good.pow(2).sum(dim=1).sqrt().pow(2)
        dist_min = (dist_good).sum()
        #dist_min = (dist_good * good_weight).sum()
        
        # far
        diff_bad = bad-avg_good
        dist_bad = diff_bad.pow(2).sum(dim=1).sqrt()
        #dist_max = dist_bad.sum()
        dist_max = torch.log2(1+dist_bad).sum()
        

        # near, min radius
        distances = torch.cdist(good, good).pow(2)
        mask = torch.eye(good.shape[0], device=good.device).bool()
        distances = distances.masked_fill(mask, float('inf'))
        # Calculate the loss for pairs that are too close
        overlap_loss = torch.relu(0.1 - distances).sum()

        # Move centers away ( good vs bad )
        center_diff = (avg_good - avg_bad).pow(2).sum()
        center_loss = center_diff #* torch.log2(1+center_diff)

        loss = dist_min - (dist_max * 10) #- (center_loss * 2) #+ (overlap_loss * 0.1) 

        #loss = dist_min - dist_max
        loss.backward()
        

        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()



### Legacy code

    def step_old(self, actives=None, weights=None):
        # Layer * Batch * Hidden * Space
        if actives is None or weights is None:
            actives, weights = self.last_actives()
        
        self.add_history()

        goods = []
        bads = []
        weights = []


        for position, active in zip(self.positions, actives):
            position_batch = position.expand( active.shape[0], -1, -1 ) 
            active_idxs = active>0

            good = position_batch[active_idxs] 
            bad = position_batch[~active_idxs] 
            goods.append(good)
            bads.append(bad)
            weight = active[active_idxs].unsqueeze(1)
            weights.append(weight)

        good = torch.vstack(goods)
        bad = torch.vstack(bads)
        weight = torch.vstack(weights)
        weight_rel = weight / weight.sum(dim=0)
        avg = torch.mean(good, dim=0).detach()

        if self.noise:
            #good += torch.randn_like(good)
            #bad += torch.randn_like(bad)
            #avg += torch.randn_like(avg) 
            pass

        #print("AVG position bad", position_batch[~active].abs().mean().item())
        
        # near
        diff_good = good-avg
        dist_good = diff_good.pow(2).sum(dim=1).sqrt()
        dist_min = (dist_good * weight_rel).sum()
        #weight_good = torch.exp(-dist_good) #.detach())
        #dist_min = (dist_good * weight_good).sum()
        
        #dist_min = torch.log(1+0.1+dist_good).sum()
        
        # far
        diff_bad = bad-avg
        dist_bad = diff_bad.pow(2).sum(dim=1).sqrt()
        dist_max = dist_bad.sum()
        #dist_max = torch.log(dist_bad+1).sum()

        #epsilon = 1e-6  # Small constant to avoid log(0)
        #weight_bad = 1 / dist_bad.detach().pow(2)
        
        #weight_bad = dist_bad.detach() + epsilon)
        
        #dist_max = (1 / dist_bad).sum()

        #weight_bad = 1 / (dist_bad.detach() )

        

        #dist_max = -(dist_bad.sum())
        
        
        
        # Calculate the loss for pairs that are too close
        #all_positions = torch.vstack(self.positions)
        #distances = torch.cdist(all_positions, all_positions)
        # Create a mask to ignore self-distances
        #mask = torch.eye(all_positions.shape[0], device=all_positions.device).bool()
        
        #distances = distances.masked_fill(mask, (self.min_dist + self.max_dist) / 2)
    
        # Calculate the loss for pairs that are too close
        #overlap_loss = torch.relu(self.min_dist - distances).sum()
        
        # Calculate the loss for pairs that are too far
        #max_distance_loss = torch.relu(distances - self.max_dist).sum()
        
        # MIN DIST ONLY
        #distances = distances.masked_fill(mask, float('inf'))
        # Calculate the loss for pairs that are too close
        #overlap_loss = torch.relu(self.min_dist - distances).sum()
        
        #reg_loss = F.l1_loss(all_positions, torch.zeros_like(all_positions))
        loss = dist_min - dist_max #* 10 + overlap_loss * 10  #+ reg_loss * 10
        loss.backward()
        
        self.optim.step()

        self.optim.zero_grad()
        


        #loss = overlap_loss * 10 # + max_distance_loss 
        #loss.backward()
        #self.optim.step()

        #self.optim.zero_grad()

        #with torch.no_grad():
        #    for self.position in self.positions:
        #        self.position.data = torch.nn.functional.normalize(self.position.data)



        # Normalize position_batch along the last dimension
        # norm_min = position_batch.min().detach()
        # norm_max = position_batch.max().detach()
        # norm_range = norm_max - norm_min
        # position_batch = (position_batch - norm_min) / norm_range
        #position_batch = nn.functional.normalize(position_batch, dim=0)
        # position_batch += torch.rand_like(position_batch).detach()
        # print(f"MIN:", dist_min.item(), 'MAX:', -dist_max.item())


        #self.optim.zero_grad()

