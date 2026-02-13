if True==False: from modularnet.modularspace import ModularSpace
if True==False: from modularnet.modularobserver import ModularObserver

import torch
from torch import nn
from torch import Tensor
import math
from utils.utils import info_tensor
import einops

class ModularModelLoss(nn.Module):
    def __init__(self, space:'ModularSpace', quantile_x=0.9, eps = 1e-8, debug = False):
        super().__init__()
        self.space = space
        self.debug = debug
        self.eps = eps

        self.reset()

    def reset(self):
        self.last_loss = None

    def forward(self):
        self.reset()
        mos = self.space.mos 
        

        layer_losses = [mo.criterion for mo in mos]
        layer_size = [loss  for loss in layer_losses]
        last_losses = [ll.last_loss for ll in layer_losses]
        goods = [layer_loss.last_center_on for layer_loss in layer_losses]
            
        good = torch.vstack(goods).reshape(len(goods), -1, self.space.space_size)
        good = einops.rearrange(good, ' b l s -> l b s')
        if self.debug:info_tensor(good, "good")
        good_centers = torch.mean(good, dim=1).unsqueeze(1)#.detach()
        if self.debug:info_tensor(good_centers, "good_centers")
        good_dist = (good - good_centers + self.eps).pow(2).sum(dim=1).sqrt()
        if self.debug:info_tensor(good_dist, "good_dist")
        good_loss = good_dist.sum(dim=1)
        last_loss = torch.vstack(last_losses)
        if self.debug:info_tensor(last_loss, "layer_loss")

        self.last_loss = last_loss.sum() + good_loss.sum() * 4

        return self.last_loss

class ModularLayerLoss(nn.Module):
    def __init__(self, observer:'ModularObserver', quantile_x=0.9, depth = 1, eps = 1e-8, debug = False):
        super().__init__()

        self.observer = observer

        self.eps = eps
        self.quantile_x = quantile_x
        self.depth = depth
        
        self.debug = debug
        self.reset()


    def reset(self):
        self.last_x = None
        self.last_q = None
        self.last_bs = None
        self.last_idxs_on = None
        self.last_idxs_off = None
        self.last_vals_on = None  
        self.last_vals_off = None
        self.last_vals_on = None
        self.last_vals_off = None
        self.last_center_on = None
        self.last_center_off = None
        self.last_center_diff = None
        self.last_center_dist = None
        self.last_weight_on = None
        self.last_weight_off = None
        self.last_dist_on = None
        self.last_dist_off = None
        self.last_loss = None

    
        
    def forward(self, x: Tensor):
        self.reset()
        
        self.last_x = x.detach()
        self.last_bs = self.last_x.shape[0]

        
        q = self.get_threashold_active(self.last_x)
        idxs_on, idxs_off = self.split_active_group(self.last_x, q)

        

        self.last_vals_on_raw = (self.last_x*idxs_on)
        self.last_vals_off_raw = -((self.last_x-q)*idxs_off)

        self.last_idxs_on = idxs_on.unsqueeze(2)
        self.last_idxs_off = idxs_off.unsqueeze(2)
        if self.debug: self.count_active()


        #TODO: normalize by the batch, not by the sample
        self.last_vals_on = (self.last_vals_on_raw / (self.last_vals_on_raw.sum(dim=1).unsqueeze(1)+self.eps)).unsqueeze(2)
        self.last_vals_off = (self.last_vals_off_raw / (self.last_vals_off_raw.sum(dim=1).unsqueeze(1)+self.eps)).unsqueeze(2)

        if self.debug:
            info_tensor(self.last_vals_on, f"{self.depth}_vals_on")
            info_tensor(self.last_vals_off, f"{self.depth}_vals_off")
            info_tensor(self.last_vals_on_raw, f"{self.depth}_vals_on_raw")
            info_tensor(self.last_vals_off_raw, f"{self.depth}_vals_off_raw")
        
        batch_positions = self.observer.positions.expand( self.last_bs, -1, -1 )

        self.last_center_on = (batch_positions * self.last_vals_on).sum(dim=1).unsqueeze(1)
        self.last_center_off = (batch_positions * self.last_vals_off).sum(dim=1).unsqueeze(1)

        
        #self.last_center_on = self.last_center_on.squeeze(1)
        #self.last_center_off = self.last_center_off.squeeze(1)
        
        #self.center_std = torch.std(center_on - (batch_positions * vals_on).sum(dim=1).unsqueeze(1))
        
        if self.debug:  
            info_tensor(self.last_center_on, f"{self.depth}_center_on")
            info_tensor(self.last_center_off, f"{self.depth}_center_off")
        self.last_center_diff = (batch_positions - self.last_center_on) + self.eps
        #print(center_diff.shape)
        self.last_center_dist = self.last_center_diff.pow(2).sum(dim=2).sqrt()
        #print(center_dist.shape)
        if self.debug:
            info_tensor(self.last_center_diff, f"{self.depth}_center_diff")

        self.last_weight_on = self.last_vals_on + self.last_idxs_on + self.eps
        self.last_weight_off = self.last_vals_off + self.last_idxs_off + self.eps

        #minimize
        self.last_dist_on = (self.last_center_diff*self.last_weight_on).pow(2).sum(dim=1).sqrt()
        #maximize
        self.last_dist_off = (self.last_center_diff*self.last_weight_off).pow(2).sum(dim=1).sqrt()
        #maximize



        
        if self.observer.loss_dist_center:
            dist_center = (self.last_center_on - self.last_center_off + self.eps).pow(2).sum(dim=1).sqrt()
            if self.debug: info_tensor(dist_center, f"{self.depth}_dist_center")


        if self.observer.loss_dist_batch:
            # maximize
            self.last_dist_batch_on = torch.cdist(self.last_dist_on, self.last_dist_on, compute_mode='use_mm_for_euclid_dist')
            # Create a mask to ignore self-distances
            mask = torch.eye(self.last_dist_on.shape[0], device=self.last_dist_on.device).bool()
            
            self.last_dist_batch_on = self.last_dist_batch_on.masked_fill(mask, float('-inf'))
            self.last_dist_batch_on = torch.relu(self.last_dist_batch_on)

        if self.debug:
            info_tensor(self.last_dist_on, f"{self.depth}_dist_on")
            info_tensor(self.last_dist_off, f"{self.depth}_dist_off")
        

        self.last_dist_on_loss = self.last_dist_on.sum(dim=1)
        self.last_dist_off_loss = self.last_dist_off.sum(dim=1)
        
        self.last_dist_on_loss = torch.log10( 1+self.last_dist_on_loss )
        self.last_dist_off_loss = (torch.log(1+self.last_dist_off_loss)*10)
        
        self.last_loss = self.last_dist_on_loss - self.last_dist_off_loss

        

        if self.observer.loss_dist_center:
            self.last_dist_center_loss  = dist_center.sum()
            self.last_loss -= self.last_dist_center_loss * 0.1

        if self.observer.loss_dist_batch:
            dist_batch_on_loss = self.last_dist_batch_on.sum()
            self.last_loss -= dist_batch_on_loss * 0.01


        self.last_loss *= 1+math.log(self.depth+1)
        

        
        if self.debug:
            info_tensor(self.last_loss, f"{self.depth}_loss")
            info_tensor(self.observer.positions, f"{self.depth}_positions")
        
        return self.last_loss
    
    def get_threashold_active(self, x:Tensor):
        self.last_q = None
        act = self.observer.observed_activation
        if isinstance(act, nn.Softmax):
            self.last_q = torch.max(self.last_x, dim=1).values.unsqueeze(1).detach()
        elif act is not None:
            ax_act = act(self.last_x)
            self.last_q = torch.quantile(ax_act, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1).detach()
        else:
            self.last_q = torch.quantile(self.last_x, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1).detach()
        return self.last_q

    def split_active_group(self,x:Tensor, q=None):
        if q is None: q = self.get_threashold_active(self.last_x)
        self.last_idxs_on  = (self.last_x >= q)
        self.last_idxs_off = (~self.last_idxs_on)
        return self.last_idxs_on, self.last_idxs_off

    
    def count_active(self):
        cnt_on = self.last_idxs_on.sum(dim=1)
        cnt_off = self.last_idxs_off.sum(dim=1)
        
        if (cnt_on==0).any():
            #print('ON','\n',cnt_on,'\n', (cnt_on==0) )
            print("WARNING: No active neurons")
            self.loss = torch.zeros(self.last_bs, device=self.device)
            self.center_on = torch.zeros(self.last_bs, self.space_size, device=self.device)
            self.center_off = torch.zeros(self.last_bs, self.space_size, device=self.device)
            return self.last_x
            #raise Exception("No active neurons")

        if (cnt_off==0).any():
            #print('OFF','\n',cnt_off,'\n', (cnt_off==0))
            print("WARNING: No inactive neurons")
            self.loss = torch.zeros(self.last_bs, device=self.device)
            self.center_on = torch.zeros(self.last_bs, self.space_size, device=self.device)
            self.center_off = torch.zeros(self.last_bs, self.space_size, device=self.device)
            return self.last_x
            #raise Exception("No inactive neurons")