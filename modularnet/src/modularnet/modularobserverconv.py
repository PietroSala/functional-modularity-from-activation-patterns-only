
import torch
from utils.utils import info_tensor
from torch import nn
from torch.autograd import Function
from itertools import combinations

import torch.nn.functional as F


debug = False



class ModularizerCNN(Function):
    @staticmethod
    def forward(ctx, x, center_dist, rescale_factor=1.00): 
        #print(x.shape)
        ctx.save_for_backward(center_dist, torch.tensor(rescale_factor) )
        return x

    @staticmethod
    def backward(ctx, grad_output):
        center_dist, rescale_factor = ctx.saved_tensors
        rescale_factor = rescale_factor.item()
        
        center_dist_norm = center_dist / center_dist.max(dim=1).values.unsqueeze(1)
        center_dist_norm = center_dist_norm.unsqueeze(2).unsqueeze(3)
        center_dist_norm = torch.functional.F.sigmoid(center_dist_norm * 10)

        grad_output_scaled = (grad_output * rescale_factor) + (grad_output * (1-center_dist_norm) * (1-rescale_factor))
        
        grad_output_total = torch.abs(grad_output).sum().item()
        grad_output_diff = torch.abs(grad_output - grad_output_scaled).sum().item()
        grad_output_perc =  int((grad_output_diff/grad_output_total)*100)
        if debug:
            print('rescale_factor', rescale_factor, grad_output_perc, grad_output_diff )


        grads = [grad_output_scaled if grads else None for grads in ctx.needs_input_grad]
        
        return tuple(grads)


class ModularObserverConv(nn.Module):
    def __init__(self,  space: 'ModularSpace', input_size:int, observed:nn.Module, depth:int, device:torch.device, locality_weight:float=1):
        super().__init__()
        self.quantile_x = 0.80
        self.locality_weight = locality_weight
        self.input_size = input_size
        self.space = space
        self.space_size = self.space.space_size
        self.observed = observed
        self.device = device
        self.positions = torch.rand( (input_size, self.space_size), requires_grad=True, device=self.device)
        self.depth = depth
        self.rescale_factor = 1.00
        self.activation_layer = None # TODO find activation

        self.stats_positions_count = torch.zeros(input_size, requires_grad=False, device=self.device)
        self.stats_data_count = 0

        self.stats_dist_on = []
        self.stats_dist_center = []

        self.loss_dist_center = True
        self.loss_dist_batch = True
        
  
        #self.active_positions = None
        #self.inactive_positions = None
        #self.weight_positions = None
        torch.autograd.set_detect_anomaly(debug)

        self.metric_covariance = False
        self.metric_count = True
        self.metric_count_total = 0
        self.activations = []
        self.counts = {comb:0 for comb in combinations(range(self.input_size), 2)} 
        
        self.center_on = None
        self.center_off = None
        self.loss = None



    def value(self):
        return self.depth #math.log(1+self.depth)
    
    def save_activations(self, x:torch.Tensor):
        if not self.metric_covariance: return
        self.activations.append(x)

    def count_activations(self, idxs:torch.Tensor):
        if not self.metric_count: return
        self.metric_count_total += idxs.shape[0]
        for line in idxs:
            pos_list = line.nonzero().squeeze().tolist()
            key_list = combinations(pos_list, 2)
            for key in key_list:
                self.counts[key] += 1


    def forward(self, x): 

        if self.space.observing == False: 
            return x


        bs = x.shape[0]
        self.stats_data_count += bs
        eps = 1e-8

        ax = x.detach()
        
        
        sorted_indices, scores = self.rank_kernels_by_activation_per_sample( ax, self.locality_weight)
        self.save_activations(scores)
        # Calculate quantile threshold for each sample
        q = torch.quantile(scores, self.quantile_x, interpolation='higher', dim=1).unsqueeze(1).detach()

        idxs_on = (scores >= q)
        idxs_off = (~idxs_on)

        self.count_activations(idxs_on)

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

        vals_on = (scores*idxs_on)
        vals_off = -((scores-q)*idxs_off)

        idxs_on = idxs_on.unsqueeze(2)
        idxs_off = idxs_off.unsqueeze(2)

        vals_on = (vals_on / (vals_on.sum(dim=1).unsqueeze(1)+eps)).unsqueeze(2)
        vals_off = (vals_off / (vals_off.sum(dim=1).unsqueeze(1)+eps)).unsqueeze(2)

        info_tensor(vals_on, f"{self.depth}_vals_on")
        info_tensor(vals_off, f"{self.depth}_vals_off")

        batch_positions = self.positions.expand( bs, -1, -1 )

        center_on = (batch_positions * vals_on).sum(dim=1).unsqueeze(1)
        center_off = (batch_positions * vals_off).sum(dim=1).unsqueeze(1)
        info_tensor(center_on, f"{self.depth}_center_on")
        info_tensor(center_off, f"{self.depth}_center_off")
        center_diff = (batch_positions - center_on) + eps
        #print(center_diff.shape)
        center_dist = center_diff.pow(2).sum(dim=2).sqrt()
        #print(center_dist.shape)

        info_tensor(center_diff, f"{self.depth}_center_diff")

        weight_on = vals_on * 2 #+ idxs_on
        weight_off = vals_off * 2 #+ idxs_off

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
        

        dist_on_loss = dist_on.sum(dim=1)
        dist_off_loss = dist_off.sum(dim=1)
        
        
        self.loss = dist_on_loss - dist_off_loss
        

        if self.loss_dist_center:
            dist_center_loss  = dist_center.sum(dim=1)
            self.loss -= dist_center_loss * 0.5

        if self.loss_dist_batch:
            dist_batch_on_loss = dist_batch_on.sum(dim=1)
            self.loss -= dist_batch_on_loss * 0.1
        
        self.center_on = center_on.squeeze(1)
        self.center_off = center_off.squeeze(1)
        

        


        info_tensor(self.loss, f"{self.depth}_loss")
        info_tensor(self.positions, f"{self.depth}_positions")
        

        #stats

        self.stats_dist_on.append(dist_on_loss.mean().item())
        self.stats_dist_center.append(dist_center_loss.mean().item())
        self.stats_positions_count += idxs_on.squeeze(2).sum(dim=0)
        
        if self.space.modularizing:
            return ModularizerCNN.apply(x, center_dist.detach(), self.rescale_factor)
        else:
            return x
        


        
    def rank_kernels_by_activation_per_sample(  self,
                                                feature_maps: torch.Tensor,
                                                locality_weight: float = 0.5,
                                                spatial_radius: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fast GPU-optimized method to rank convolutional kernels for each sample in the batch.
        
        Args:
            feature_maps: Tensor of shape (batch_size, num_kernels, height, width)
            locality_weight: Weight factor for locality score (0 to 1)
            spatial_radius: Radius for computing local response
        
        Returns:
            sorted_indices: Indices of kernels sorted by importance for each sample
                        Shape: (batch_size, num_kernels)
            kernel_scores: Corresponding importance scores for each sample
                        Shape: (batch_size, num_kernels)
        """
        B, C, H, W = feature_maps.shape
        
        # 1. Compute Per-Sample Activation Magnitude
        # Using L2 norm for each sample separately
        magnitude_scores = torch.norm(feature_maps.reshape(B, C, -1), dim=2)  # Shape: (B, C)
        
        # 2. Compute Local Response Concentration per sample
        padding = spatial_radius // 2
        max_pooled = F.max_pool2d(
            feature_maps,
            kernel_size=spatial_radius,
            stride=1,
            padding=padding
        )
        
        # Compute ratio of local maximum to total activation for each sample
        local_response = (max_pooled.abs() / (torch.sum(feature_maps.abs(), dim=(2, 3), keepdim=True) + 1e-8))
        locality_scores = local_response.mean(dim=(2, 3))  # Shape: (B, C)
        
        # 3. Normalize scores per sample
        magnitude_scores = magnitude_scores / (magnitude_scores.max(dim=1, keepdim=True)[0] + 1e-8)
        locality_scores = locality_scores / (locality_scores.max(dim=1, keepdim=True)[0] + 1e-8)
        
        # 4. Compute final scores per sample
        combined_scores = ((1 - locality_weight) * magnitude_scores) + (locality_weight * locality_scores)
        
        # 5. Sort independently for each sample
        sorted_indices = torch.argsort(combined_scores, dim=1, descending=True)
        
        return sorted_indices, combined_scores

    # Optional: Version that returns top-k kernels per sample
    def get_top_k_kernels_per_sample(   self,
                                        feature_maps: torch.Tensor,
                                        k: int,
                                        locality_weight: float = 0.5,
                                        spatial_radius: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the top-k most important kernels for each sample.
        
        Args:
            feature_maps: Tensor of shape (batch_size, num_kernels, height, width)
            k: Number of top kernels to return
            locality_weight: Weight factor for locality score
            spatial_radius: Radius for computing local response
        
        Returns:
            top_k_indices: Indices of top-k kernels per sample
                        Shape: (batch_size, k)
            top_k_scores: Scores of top-k kernels per sample
                        Shape: (batch_size, k)
            top_k_features: Feature maps of top-k kernels per sample
                        Shape: (batch_size, k, height, width)
        """
        sorted_indices, scores = self.rank_kernels_by_activation_per_sample(
            feature_maps, locality_weight, spatial_radius
        )
        
        # Get top-k for each sample
        top_k_indices = sorted_indices[:, :k]  # Shape: (B, k)
        top_k_scores = torch.gather(scores, 1, top_k_indices)  # Shape: (B, k)
        
        # Get corresponding feature maps
        batch_indices = torch.arange(feature_maps.shape[0], device=feature_maps.device)
        batch_indices = batch_indices.view(-1, 1).expand(-1, k)
        top_k_features = feature_maps[batch_indices, top_k_indices]  # Shape: (B, k, H, W)
        
        return top_k_indices, top_k_scores, top_k_features
    
    def get_quantile_k_kernels_per_sample(self,
                                feature_maps: torch.Tensor,
                                quantile: float,
                                locality_weight: float = 0.5,
                                spatial_radius: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns kernels above the quantile threshold for each sample.
        
        Args:
            feature_maps: Tensor of shape (batch_size, num_kernels, height, width)
            quantile: Float between 0 and 1, threshold for kernel selection
            locality_weight: Weight factor for locality score
            spatial_radius: Radius for computing local response
            
        Returns:
            top_k_indices: Indices of selected kernels per sample
                        Shape: (batch_size, k), where k varies by quantile
            top_k_scores: Scores of selected kernels per sample
                        Shape: (batch_size, k)
            top_k_features: Feature maps of selected kernels per sample
                        Shape: (batch_size, k, height, width)
        """
        sorted_indices, scores = self.rank_kernels_by_activation_per_sample(
            feature_maps, locality_weight, spatial_radius
        )
        
        # Calculate quantile threshold for each sample
        thresholds = torch.quantile(scores, quantile, dim=1)
        
        # Create mask for selected kernels and apply to sorted indices and scores
        selected_mask = scores >= thresholds.unsqueeze(1)
        k = selected_mask.sum(dim=1).max().item()  # Get max number of selected kernels
        
        # Get top-k indices and scores using the mask
        top_k_indices = torch.zeros((feature_maps.shape[0], k), 
                                dtype=torch.long, 
                                device=feature_maps.device)
        top_k_scores = torch.zeros((feature_maps.shape[0], k), 
                                device=feature_maps.device)
        
        for b in range(feature_maps.shape[0]):
            sample_indices = torch.where(selected_mask[b])[0]
            num_selected = len(sample_indices)
            top_k_indices[b, :num_selected] = sorted_indices[b, sample_indices]
            top_k_scores[b, :num_selected] = scores[b, sample_indices]
        
        # Get corresponding feature maps
        batch_indices = torch.arange(feature_maps.shape[0], device=feature_maps.device)
        batch_indices = batch_indices.view(-1, 1).expand(-1, k)
        top_k_features = feature_maps[batch_indices, top_k_indices]
        
        return top_k_indices, top_k_scores, top_k_features