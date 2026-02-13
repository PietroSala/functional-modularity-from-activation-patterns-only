
from calendar import c
from modularnet.modularcluster import ModularClusterKmeans
from modularnet.modularloss import ModularModelLoss
from modularnet.modularobserverconv import ModularObserverConv
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from typing import *
import einops
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy

from modularnet.modularobserver import ModularObserver
from utils.utils import info_tensor, is_activation, is_conv, is_layer



class ModularSpace():
    def __init__(self, name:str, module: nn.Module, space_size:int, lr:float=0.5, history:bool=True, noise=True, useLayers=True, useActivations=True, device=None, base_path=None):
        self.name = name
        self.max_dist = 2
        self.min_dist = 0.1
        self.module = module
        self.space_size = space_size
        self.noise = noise 
        
        # Get device from module if not specified, ensuring consistency
        if device is not None:
            self.device = torch.device(device)
        else:
            # Try to infer device from the module's parameters
            try:
                model_device = next(module.parameters()).device
                self.device = model_device
            except StopIteration:
                # If module has no parameters, use default device
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.lr = lr
        self.lr_steps = [10000] # [1000,2000,3000]
        self.observing = True
        self.modularizing = True
        self.mos = [] #type: List[ModularObserver]
        self.rescale_gamma = 0.70
        self.rescale_factor = self.rescale_gamma
        self.space_loss = ModularModelLoss(self)

        self.base_path = './' if base_path is None else base_path
        
        #print('summary model', summary(self.module,(784,)))
        self.inject_observer(self.module, useLayers=useLayers, useActivations=useActivations)
        #print('summary model injected', summary(self.module,(784,)))
        # Layer * 1 * Hidden * Space 
        #self.mos = [module for module in self.mos] #type: List[ModularObserver]
        #print("SOS: ",[(sos.depth,sos.input_size) for sos in self.sos])
        self.positions = [mo.positions for mo in self.mos]
        
        # Ensure all positions are on the correct device
        self.positions = [pos.to(device=self.device) for pos in self.positions]

        self.optim = torch.optim.AdamW(self.positions, lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=self.lr_steps, gamma=0.5)

        self.history = history
        

        
    
        self.reset()

    def reset(self):
        self.position_history = []

        self.last_cluster_centers = None
        self.last_clustering = None
        self.last_clustering_all = None
        self.last_cluster_stats = {}
        self.last_cluster_stats_wrong = {}

    def path_checkpoint(self, filename):
        basepath = self.base_path + '/' if self.base_path is not None else ''
        basepath = os.path.join(basepath,'checkpoint')
        os.makedirs(basepath, exist_ok=True)
        path = os.path.join(basepath, filename)
        return path


    def save_cluster_map(self):
        path = self.path_checkpoint(f'{self.name}_clusters.pth')
        if os.path.exists(path): os.remove(path)
        clustering_data = (self.last_cluster_centers,self.last_clustering_all, self.last_clustering)
        torch.save(clustering_data, path) 
    
    def load_cluster_map(self):
        path = self.path_checkpoint(f'{self.name}_clusters.pth')
        if not os.path.exists(path): return False
        clustering_data = torch.load(path, weights_only=True)
        self.last_cluster_centers, self.last_clustering_all, self.last_clustering = clustering_data
        self.last_clustering_all = self.last_clustering_all.to(device=self.device)
        self.last_clustering = [clustering.to(device=self.device) for clustering in self.last_clustering]
        return True
    
    def save_cluster_stats(self):
        path = self.path_checkpoint(f'{self.name}_clusters_stats.pth')
        if os.path.exists(path): os.remove(path)
        clustering_data = (self.last_cluster_stats, self.last_cluster_stats_wrong)
        torch.save(clustering_data, path) 
    
    def load_cluster_stats(self):
        path = self.path_checkpoint(f'{self.name}_clusters_stats.pth')
        if not os.path.exists(path): return False
        clustering_data = torch.load(path, weights_only=True)
        self.last_cluster_stats, self.last_cluster_stats_wrong = clustering_data
        #self.last_cluster_stats = [clustering.to(device=self.device) for clustering in self.last_cluster_stats]
        #self.last_cluster_stats_wrong = [clustering.to(device=self.device) for clustering in self.last_cluster_stats_wrong]
        return True

    def save_space(self, path=None):
        path = self.path_checkpoint(f'{self.name}_space.pth')
        if os.path.exists(path): os.remove(path)
        torch.save(self.positions, path)

    def load_space(self, path=None):
        path = self.path_checkpoint(f'{self.name}_space.pth')
        if not os.path.exists(path): return False
        space_data = torch.load(path, weights_only=True)
        for pos, data  in zip(self.positions, space_data):
            pos.data += data.data 
        return True

    def is_observer(self,module):
        if isinstance(module, ModularObserver): return True
        if isinstance(module, ModularObserverConv): return True
        return 

    def inject_observer(self, parent: nn.Module, out_features:int|None=None, depth=1, useLayers=True, useActivations=True):
        for name, child in parent.named_children():
            # child.register_full_backward_hook(self.backward_hook)
            if hasattr(child,'out_features'): out_features = child.out_features
            elif hasattr(parent,'out_features'): out_features = parent.out_features

            if (useActivations and is_activation(child)):
                self.mos[-1].observed_activation = child
            elif (useLayers and is_layer(child)):
                #print(name)

                if is_conv(child):
                    out_features = child.out_channels
                    mo = ModularObserverConv(self, out_features, child, depth, self.device)
                else:
                    mo = ModularObserver(self, out_features, child, depth, self.device)
                self.mos.append(mo)

                # Forward search for activation layer
                #activation_layer = self.find_activation_layer(child)
                #if activation_layer:
                #    mo.activation_layer = activation_layer
                #    print(f"Associated activation layer: {activation_layer}")

                #new_sequence = nn.Sequential( child, mo )
                #setattr(parent, name, new_sequence)
                
                depth +=1
            else:
                out_features = self.inject_observer(child, out_features, depth, useLayers, useActivations)
        
        return out_features

    def find_activation_layer(self, module: nn.Module):
        """ Recursively searches for an activation layer in the given module's children. """
        for _, child in module.named_children():
            if is_activation(child): return child
            result = self.find_activation_layer(child)  # Recursive search
            if result: return result
                
        return None  # No activation found
            
    def last_actives(self):
        actives = [mo.criterion.last_center_on for mo in self.mos]
        inactives = [mo.criterion.last_center_off for mo in self.mos]
        weights = [mo.criterion.last_loss for mo in self.mos]
        return actives, inactives, weights

    def last_centers(self):
        actives = [mo.criterion.last_center_on for mo in self.mos]
        inactives = [mo.criterion.last_center_off for mo in self.mos]
        losses = [mo.criterion.last_loss for mo in self.mos]
        return actives, inactives, losses

    def clear_history(self):
        self.position_history = []

    def add_history(self):
        if (self.history):
            pos = [ pos.clone().tolist() for pos in self.positions]
            self.position_history.append( pos )
    
    def rescale_factor_step(self):
        self.rescale_factor = self.rescale_factor * self.rescale_gamma
        for mos in self.mos:
            mos.rescale_factor = self.rescale_factor

    
    

    def rate(self):
        
        actives = [mo.criterion.last_center_on for mo in self.mos]
        
        #self.add_history()
        
        stats_std = []
        stats_avg = []
        
        bs = len(actives[0])
    
        for i in range(bs):
            goods = []

            for active in actives:
                goods.append(active[i])

            good = torch.vstack(goods).detach()
            
            #unweighted
            std, avg = torch.std_mean(good, dim=0)

            
            
            
            stats_std.append(std.tolist())
            stats_avg.append(avg.tolist())
        
        return stats_avg, stats_std
    
    def cluster_space(self, method='kmeans', plot=False, save=True):
        #positions = torch.load(basepath + 'fashion_mnist_cnn_space.pth', weights_only=True)
        pos_cnts = tuple([len(pos) for pos in self.positions ])
        positions_all = torch.concatenate(self.positions, dim=0)
        
        print("clusterspace: positions_all: ", positions_all.shape)
        #print(positions.shape)
        positions_all = positions_all.detach().cpu().numpy()

        
        mc_kmeans = ModularClusterKmeans()
        #mc_dbscan = ModularClusterDBScan()
        #mc_hdbscan = ModularClusterHDBScan() 

        ## KMEANS + elbow
        lbl_kmeans, cluster_centers = mc_kmeans.cluster_kmeans(positions_all)
        #scores_kmeans = mc_kmeans.cluster_scores(positions_all, lbl_kmeans)
        if plot: mc_kmeans.plot_cluster(positions_all, lbl_kmeans, "KMEANS")
        
        self.last_cluster_centers = torch.from_numpy(cluster_centers).to(device=self.device)
        self.last_clustering_all = torch.from_numpy(lbl_kmeans).to(device=self.device) + 1
        self.last_clustering = self.last_clustering_all.split( pos_cnts )
        #clusters_label = torch.unique(self.last_clustering_all)
        #self.last_cluster_stats = {lbl.item():{} for lbl in clusters_label}
        if save: self.save_cluster_map()
        return self.last_clustering  #, clusters_label
    


    def cluster_class_stats(self):
        if self.last_cluster_stats is None:
            print("No cluster stats available.")
            return


        

        clusters_sum = {} 
        classes = set()
        for cluster_num, classes_stats in self.last_cluster_stats.items():
            cluster_num = int(cluster_num)
            class_sum = {}
            for c, cnt in classes_stats.items():
                if c not in class_sum: class_sum[c] = 0
                class_sum[c] += cnt
                classes.add(c)
            clusters_sum[cluster_num] = class_sum   
        
        classes = list(sorted(classes))
        table_header = ['Nr'] + classes
        table_data = []
        for cluster_num, class_sum in clusters_sum.items():
            # Ensure all classes are present in each cluster
            #for c in classes:
            #    if c not in class_sum:
            #        class_sum[c] = 0
            row = [f"#{cluster_num}"] + [class_sum.get(c, 0) for c in classes]
            table_data.append(row)
        cluster_stats = pd.DataFrame(table_data, columns=table_header)
        #print(cluster_stats)
        row_classes_hits = cluster_stats[classes]
        # Compute standard uniform distribution for each row

        uniform_dist = np.ones(len(row_classes_hits.columns)) / len(row_classes_hits.columns)
        # Compute KL divergence for each cluster (row)
        kl_divergences = []
        for idx, row in row_classes_hits.iterrows():
            p = row.values
            p_sum = p.sum()
            if p_sum == 0:
                kl = 0.0
            else:
                p_norm = p / p_sum
                kl = entropy(p_norm, uniform_dist)
            kl_divergences.append(kl)
        kld = np.array(kl_divergences)
        cluster_stats['KL_div'] = kld

        kld_exp = 1-np.exp(-kld) #exponential projection
        cluster_stats['KL_exp'] = kld_exp


        conditions = [
            (kld_exp < 0.25),
            (kld_exp >= 0.25) & (kld_exp <= 0.6),
            (kld_exp > 0.6)
        ]
        choices = np.array(['GENERAL', 'STANDARD', 'SPECIALIZED'])

        # Add the new column
        kl_type = np.select(conditions, choices, default='---')
        cluster_stats['kl_type'] = kl_type


        
        # top classes
        rows = []
        for idx, row in row_classes_hits.iterrows():
            normalized_row = row / row.sum()

            # Sort classes in descending order and compute cumulative sum
            sorted_classes = normalized_row.sort_values(ascending=False)
            cumsum = sorted_classes.cumsum()
            
            # Find the classes that make up the 90% quantile
            top_classes = sorted_classes[cumsum <= 0.90]
            if len(top_classes) == 0:  # Handle case where a single class is >90%
                top_classes = sorted_classes.head(1)
            rows.append(tuple(top_classes.index.tolist()))
            #print(top_classes.index.tolist())
        cluster_stats['top_classes'] = tuple(rows)

        # count neurons:
        class_count = np.unique_counts(self.last_clustering_all.cpu().numpy())
        class_count = {f'#{num}':int(cnt) for num,cnt in zip(*class_count)}
        cluster_stats['CU cnt'] = cluster_stats['Nr'].apply(lambda x: class_count.get(x, 0))
        
        # compute distances
        centers_stats = False
        if centers_stats:
            centers = self.last_cluster_centers
            centers_abs = centers.abs().to(dtype=torch.float64)
            centers_min = centers_abs.amin(axis=1).cpu().numpy()
            centers_max = centers_abs.amax(axis=1).cpu().numpy()
            centers_mean = centers_abs.mean(axis=1).cpu().numpy()
            centers_std = centers_abs.std(axis=1).cpu().numpy()
            centers_dist = centers_abs.pow(2).mean(axis=1).sqrt().cpu().numpy()


            cluster_stats['dist min'] = centers_min
            cluster_stats['dist max'] = centers_max
            cluster_stats['dist mean'] = centers_mean
            cluster_stats['dist std'] = centers_std
            cluster_stats['dist euc'] = centers_dist

        

        row_totals = row_classes_hits.sum(axis=1)
        row_totals_sum = row_totals.sum()
        #print(row_totals_sum)
        
        row_totals_perc = row_totals.div(row_totals_sum).mul(100).round(2)
         
        
        #print(row_totals)
        
        #print(cluster_stats)

        #classes_prec = [f'{c} %' for c in classes]
        series_cls_perc = cluster_stats[classes]
        #print(series_cls_perc)
        series_cls_perc = series_cls_perc.div(row_totals, axis=0).mul(100).round(2)
        #print(series_cls_perc)
        cluster_stats[classes] = series_cls_perc

        row_totals_std = series_cls_perc.std(axis=1).round(2)

        cluster_stats['Total'] = row_totals
        cluster_stats['Total (%)'] =  row_totals_perc
        cluster_stats['Std'] =  row_totals_std

        cluster_stats = cluster_stats.round(2)

        print(cluster_stats)
        cluster_stats.to_csv(f"{self.base_path}/{self.name}_cluster_class_attribution.csv", index=False)
        return cluster_stats, classes

    def class_cluster_attribution(self, ys, ys_hat):
        #print("class_cluster_attribution")
        actives, _, _ = self.last_centers()
        last_idxs_on = [mo.criterion.last_idxs_on for mo in self.mos]
        last_vals_on = [mo.criterion.last_vals_on for mo in self.mos]
        
        last_clusters_on = [ cluster_labels.repeat(idx_on.shape[0]).reshape(idx_on.shape[0],-1) * idx_on.squeeze(2)  for cluster_labels, idx_on in zip(self.last_clustering, last_idxs_on)]
        last_clusters_count = [ [torch.unique(sample[sample>0], sorted=False ,return_counts=True) for sample in cluster_labels]  for cluster_labels in last_clusters_on]
        #print("last_clusters_on", len(last_clusters_on))
        #print("last_clusters_count", len(last_clusters_count))
        clusters_count_batch = []

        for batch_j in range(len(last_clusters_count[0])):
            line = []
            for layer_i in range(len(last_clusters_count)):
                line.append( torch.vstack(last_clusters_count[layer_i][batch_j]) )
            clusters_count_batch.append( torch.cat(line, dim=1) )
            
        for y, y_hat, (cluster_labels, hit_counts) in zip(ys, ys_hat, clusters_count_batch):
            stats = self.last_cluster_stats if y == y_hat else self.last_cluster_stats_wrong
            for cluster_label, hit_count in zip(cluster_labels, hit_counts):
                cluster_label = cluster_label.item()
                if cluster_label not in stats: stats[cluster_label] = {}
                key = f'{y}' if y == y_hat else f'{y}_{y_hat}'
                if key not in stats[cluster_label]: stats[cluster_label][key] = 0
                stats[cluster_label][key] += hit_count.item()
    
    def set_pruning_masks(self, masks:List[torch.Tensor]=None):
        if masks is None: masks = []
        
        for mo, mask in zip(self.mos, masks):
            if mo == self.mos[-1]: print("set_pruning_masks: skip last layer"); continue
            if mask is None: print("set_pruning_masks: empty mask"); continue
            if not isinstance(mask, torch.Tensor): print("set_pruning_masks: not a tensor!"); continue
            if len(mask) != len(mo.positions): print("set_pruning_masks: size mismatch!"); continue
            mo.set_pruning_mask(mask)

    def mask_by_class(self, class_num, threshold=0.0):# -> list[Tensor | Any]:
        masks = [torch.zeros_like(ls) for ls in self.last_clustering]
        for cluster_num, classes_num in self.last_cluster_stats.items():
            class_vals = np.array(list(classes_num.values()))
            total = np.sum(class_vals)
            
            filtered_classed = [k for k,v in classes_num.items() if total > 0 and v/total >= threshold]

            if str(class_num) in filtered_classed: continue
            cluster_layer_masks = [(layer_clustering == cluster_num) for layer_clustering in self.last_clustering]
            masks = [mask + clm for mask, clm in zip(masks,cluster_layer_masks)]
        masks = [(mask <= 0) for mask in masks] # invert bool mask: 0 = True, >0 = False
        
        #print(masks)

        #cns = [ ( (m == False).sum().item(),m.numel() ) for m in masks]
        #print(class_num, cns)

        return masks
        
    
    def step(self, actives=None, inactives=None, weights=None):
        
        # Layer * Batch * Hidden * Space
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





    def step_batch(self, history=True):
        if not self.observing: return
        loss = self.space_loss.forward()
        loss.backward()
        


        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()
        
        if history: self.add_history()



    
    def step_batch_old(self, centers_on=None, centers_off=None, losses=None, history=True):
        if self.observing == False: return
        
        # Layer * Batch * Hidden * Spac5e
        if centers_on is None or centers_off is None or losses is None:
            centers_on, centers_off, losses = self.last_centers()

   
        goods = []
        layer_losses = []
        
        for active, inactive, layer_loss in zip(centers_on, centers_off, losses):
            goods.append(active)
            layer_losses.append(layer_loss)
            
        good = torch.vstack(goods).reshape(len(goods), -1, self.space_size)
        good = einops.rearrange(good, ' b l s -> l b s')
        info_tensor(good, "good")
        good_centers = torch.mean(good, dim=1).unsqueeze(1).detach()
        info_tensor(good_centers, "good_centers")
        good_dist = (good - good_centers).pow(2).sum(dim=1).sqrt()
        info_tensor(good_dist, "good_dist")
        good_loss = good_dist.sum(dim=1)
        layer_loss = torch.vstack(layer_losses)
        info_tensor(layer_loss, "layer_loss")

        loss = layer_loss.sum() + good_loss.sum() * 4
        loss.backward()
        


        self.optim.step()
        self.lr_scheduler.step()
        self.optim.zero_grad()
        
        if history: self.add_history()


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


    def rate_old(self, actives=None, weights=None):
        # Layer * Batch * Hidden * Space
        if actives is None or weights is None:
            actives, inactives, losses = self.last_centers()
        
        self.add_history()
        
        stats_std = []
        stats_avg = []
        
        bs = len(actives[0])
    
        for i in range(bs):
            goods = []
            
            #good_weights=[]

            for active, weight in zip(actives, weights):
                goods.append(active[i])
            #    good_weights.append(weight[i])

            good = torch.vstack(goods).detach()
            #good_weight = torch.concat(good_weights)
            #good_weight = (good_weight / good_weight.sum()).unsqueeze(1)
            

            #unweighted
            std, avg = torch.std_mean(good, dim=0)

            #weighted
            #avg = torch.mean(good, dim=0).detach()
            #avg = torch.sum(good * good_weight, dim=0).detach()
            #std = torch.mean( (avg-good).pow(2), dim=0 ).detach()

            
            
            stats_std.append(std.tolist())
            stats_avg.append(avg.tolist())
        
        return stats_avg, stats_std