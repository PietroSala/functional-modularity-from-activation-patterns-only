if True==False: from modularnet.modularspace import ModularSpace

from modularnet.modularloss import ModularLayerLoss
import torch
from torch import Tensor
from utils.utils import info_tensor
from torch import nn
from itertools import combinations

debug = True

""" deprecated for hooks 
class ModularizerLinear(Function):
    @staticmethod
    def forward(ctx, x, center_dist, rescale_factor=1.00): 
        #print(x.shape)
        ctx.save_for_backward(center_dist, torch.tensor(rescale_factor) )
        return x

    @staticmethod
    def backward(ctx, grad_output):
        center_dist, rescale_factor = ctx.saved_tensors
        rescale_factor = rescale_factor.item()
        
        center_dist = torch.pow(center_dist, 2)
        center_dist_norm = center_dist / center_dist.max(dim=1).values.unsqueeze(1)
        center_dist_norm = torch.functional.F.sigmoid(center_dist_norm * 10)

        grad_output_scaled = (grad_output * rescale_factor) + (grad_output * (1-center_dist_norm) * (1-rescale_factor))
        
        grad_output_total = torch.abs(grad_output).sum().item()
        grad_output_diff = torch.abs(grad_output - grad_output_scaled).sum().item()
        grad_output_perc =  int((grad_output_diff/grad_output_total)*100)
        if debug:
            print('rescale_factor', rescale_factor, grad_output_perc, grad_output_diff )
z

        grads = [grad_output_scaled if grads else None for grads in ctx.needs_input_grad]
        
        return tuple(grads)
"""

class ModularObserver():
    def __init__(self,  space: 'ModularSpace', input_size:int, observed:nn.Module, depth:int, device:torch.device, debug=False, auto_hook=True):
        self.quantile_x = 0.80
        self.rescale_factor = 1.00
        self.loss_dist_center = True
        self.loss_dist_batch = True
        self.device = device
        
        
        # space
        self.input_size = input_size
        self.space = space
        self.space_size = self.space.space_size
        self.positions = torch.rand( (input_size, self.space_size), requires_grad=True, device=self.device)
        self.depth = depth
        self.pruning_mask = None 
        
        #model layers
        self.observed = observed
        self.observed_activation = None
        self.criterion = ModularLayerLoss(self)
         
        # debug
        self.debug = debug
        self.plot_activations = False
        torch.autograd.set_detect_anomaly(True)


        # stats
        self.stats_positions_count = torch.zeros(input_size, requires_grad=False, device=self.device)
        self.stats_data_count = 0

        self.stats_dist_on = []
        self.stats_dist_center = []

        
        # metrics
        self.metric_covariance = False
        self.metric_count = True
        self.metric_count_total = 0
        self.activations = []
        self.counts = {comb:0 for comb in combinations(range(self.input_size), 2)} 

        # hooks
        self.forward_hook = None
        self.backward_hook = None

        self.reset()
        if auto_hook: self.enable_hooks()
        

    def reset(self):
        self.last_loss = None

        
    def enable_hooks(self):
        self.disable_hooks()
        self.forward_hook = self.observed.register_forward_hook(self.forward_callback)
        #self.backward_hook = self.observed.register_full_backward_hook(self.backward_callback)
        self.backward_hook = self.observed.register_full_backward_pre_hook(self.backward_callback)
    
    def disable_hooks(self):
        if self.forward_hook is not None: 
            self.forward_hook.remove()
            self.forward_hook = None
        
        if self.backward_hook is not None:
            self.backward_hook.remove()
            self.backward_hook = None

    def value(self):
        return self.depth #math.log(1+self.depth)
    
    def save_activations(self, x:torch.Tensor):
        if not self.metric_covariance: return
        self.activations.append(x)

    def count_activations(self, idxs:torch.Tensor):
        if not self.metric_count: return
        self.metric_count_total += idxs.shape[0]
        for line in idxs:
            pos_list = line.nonzero()
            pos_list = pos_list.squeeze(dim=1)
            pos_list = pos_list.tolist()
            key_list = list(combinations(pos_list, 2))
            for key in key_list:
                self.counts[key] += 1
    

    def set_pruning_mask(self, mask:torch.Tensor=None):
        if mask is None:
            self.pruning_mask = None
        if len(mask) != self.input_size: 
            raise ValueError(f"Mask length {len(mask)} does not match input size {self.input_size}") 

        self.pruning_mask = mask.detach().to(device=self.device)

    def forward_callback(self, layer:nn.Module, x_in:torch.Tensor, x_out:torch.Tensor):
        if not self.space.observing: return x_out
        if self.pruning_mask is not None:
            x_out *= self.pruning_mask

            return x_out

        self.last_loss = self.criterion(x_out)
        #stats
        self.save_activations(x_out) #######
        self.count_activations(self.criterion.last_idxs_on) ######

        self.stats_dist_on.append(self.criterion.last_center_diff.mean().item())
        self.stats_dist_center.append(self.criterion.last_center_dist.mean().item())
        self.stats_positions_count += self.criterion.last_idxs_on.squeeze(2).sum(dim=0)

        #return layer_loss


    def backward_callback(self, layer:nn.Module, grads_out:tuple[Tensor]):
        if not self.space.modularizing: return grads_out
        
        eps = 1e-8
        center_dist = self.criterion.last_center_dist

        center_dist = torch.pow(center_dist+eps, 2)
        center_dist_norm = center_dist / (center_dist.amax(dim=1).unsqueeze(1) + eps )
        center_dist_norm = torch.functional.F.sigmoid(center_dist_norm * 10)
        alpha = self.rescale_factor
        rescale_grads = lambda g_in, alpha: (g_in * alpha) + (g_in * (1-center_dist_norm) * (1-alpha))
        grad_output_scaled = [None if grad_out is None else rescale_grads(grad_out,alpha) for grad_out in grads_out]
        #grad_input_scaled = []

        #for g_in in grads_out:
        #    grad_input_scaled.append( (g_in * a) + (g_in * (1-center_dist_norm) * (1-a))  )
        
        #print(grads_out)
        #print(layer)
        grad_output_total = [torch.abs(grad_out).sum().item() for grad_out in grads_out]
        grad_output_diff = [torch.abs(grad_out - scaled).sum().item() for grad_out, scaled in zip(grads_out, grad_output_scaled) ]
        grad_output_perc =  [int((diff/total)*100) for diff, total in zip(grad_output_diff, grad_output_total) ]
        
        if debug and False:
            print('rescale_factor', self.rescale_factor, grad_output_perc, grad_output_diff )


        return grad_output_scaled


    def forward_old(self, x): 

        if self.space.observing == False: 
            return x

        
        bs = x.shape[0]
        self.stats_data_count += bs
        eps = 1e-8

        ax = x.detach()
        
        self.save_activations(ax) #######

        
        
        act = self.observed_activation
        if isinstance(act, nn.Softmax):
            q = torch.max(ax, dim=1).values.unsqueeze(1).detach()
        elif act is not None:
            ax_act = act(ax)
            q = torch.quantile(ax_act, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1).detach()
        else:
            q = torch.quantile(ax, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1).detach()
        
        #q = 0

        idxs_on = (ax >= q)
        idxs_off = (~idxs_on)

        self.count_activations(idxs_on) ######

        cnt_on = idxs_on.sum(dim=1)
        cnt_off = idxs_off.sum(dim=1)
        
        if (cnt_on==0).any():
            #print('ON','\n',cnt_on,'\n', (cnt_on==0) )
            print("WARNING: No active neurons")
            self.loss = torch.zeros(bs, device=self.device)
            self.center_on = torch.zeros(bs, self.space_size, device=self.device)
            self.center_off = torch.zeros(bs, self.space_size, device=self.device)
            return x
            #raise Exception("No active neurons")

        if (cnt_off==0).any():
            #print('OFF','\n',cnt_off,'\n', (cnt_off==0))
            print("WARNING: No inactive neurons")
            self.loss = torch.zeros(bs, device=self.device)
            self.center_on = torch.zeros(bs, self.space_size, device=self.device)
            self.center_off = torch.zeros(bs, self.space_size, device=self.device)
            return x
            #raise Exception("No inactive neurons")

        vals_on = (ax*idxs_on)
        vals_off = -((ax-q)*idxs_off)

        idxs_on = idxs_on.unsqueeze(2)
        idxs_off = idxs_off.unsqueeze(2)

        vals_on = (vals_on / (vals_on.sum(dim=1).unsqueeze(1)+eps)).unsqueeze(2)
        vals_off = (vals_off / (vals_off.sum(dim=1).unsqueeze(1)+eps)).unsqueeze(2)

        info_tensor(vals_on, f"{self.depth}_vals_on")
        info_tensor(vals_off, f"{self.depth}_vals_off")

        batch_positions = self.positions.expand( bs, -1, -1 )

        center_on = (batch_positions * vals_on).sum(dim=1).unsqueeze(1)
        center_off = (batch_positions * vals_off).sum(dim=1).unsqueeze(1)

        self.center_on = center_on.squeeze(1)
        self.center_off = center_off.squeeze(1)
        
        #self.center_std = torch.std(center_on - (batch_positions * vals_on).sum(dim=1).unsqueeze(1))
        
          
        info_tensor(center_on, f"{self.depth}_center_on")
        info_tensor(center_off, f"{self.depth}_center_off")
        center_diff = (batch_positions - center_on) + eps
        #print(center_diff.shape)
        center_dist = center_diff.pow(2).sum(dim=2).sqrt()
        #print(center_dist.shape)

        info_tensor(center_diff, f"{self.depth}_center_diff")

        weight_on = vals_on + idxs_on
        weight_off = vals_off + idxs_off

        #minimize
        dist_on = (center_diff*weight_on).pow(2).sum(dim=1).sqrt()
        #maximize
        dist_off = (center_diff*weight_off).pow(2).sum(dim=1).sqrt()
        #maximize

        
        

        
        if self.loss_dist_center:
            dist_center = (center_on - center_off).pow(2).sum(dim=1).sqrt()
            info_tensor(dist_center, f"{self.depth}_dist_center")


        if self.loss_dist_batch:
            # maximize
            dist_batch_on = torch.cdist(dist_on, dist_on, compute_mode='use_mm_for_euclid_dist')
            # Create a mask to ignore self-distances
            mask = torch.eye(dist_on.shape[0], device=dist_on.device).bool()
            
            dist_batch_on = dist_batch_on.masked_fill(mask, float('-inf'))
            dist_batch_on = torch.relu(dist_batch_on)


        info_tensor(dist_on, f"{self.depth}_dist_on")
        info_tensor(dist_off, f"{self.depth}_dist_off")
        

        dist_on_loss = torch.log10( dist_on ).sum(dim=1)
        dist_off_loss = (torch.log(dist_off)*10).sum(dim=1)
        
        
        self.loss = dist_on_loss - dist_off_loss

        

        if self.loss_dist_center:
            dist_center_loss  = dist_center.sum(dim=1)
            self.loss -= dist_center_loss * 0.1

        if self.loss_dist_batch:
            dist_batch_on_loss = dist_batch_on.sum(dim=1)
            self.loss -= dist_batch_on_loss * 0.01


        self.loss *= self.depth
        

        


        info_tensor(self.loss, f"{self.depth}_loss")
        info_tensor(self.positions, f"{self.depth}_positions")
        

        #stats

        self.stats_dist_on.append(dist_on_loss.mean().item())
        self.stats_dist_center.append(dist_center_loss.mean().item())
        self.stats_positions_count += idxs_on.squeeze(2).sum(dim=0)
        
        if self.space.modularizing:
            return ModularizerLinear.apply(x, center_dist.detach(), self.rescale_factor)
        else:
            return x