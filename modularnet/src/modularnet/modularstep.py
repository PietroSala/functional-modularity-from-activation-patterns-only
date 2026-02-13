
from ast import Mod
import time
from dataprovider.example_models import FashionMnistModel


from modularnet.metamodel.metamodel import BaseModel, MetaModel
from modularnet.metamodel.metamodelconfig import MetaModelConfig
from modularnet.modularstats import ModularStats

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from dataprovider.dataprovider import BaseDataProvider, FashionMnistDataProvider

from modularnet.modularspacevisualizer import ModularSpaceVisualizer
from modularnet.modularspace import ModularSpace
from sklearn.metrics import confusion_matrix
from utils.utils import f1s_from_truth_table, seed_everything, session_path, set_best, truth_table_from_cm

def test():
    seed = 42
    if seed is not None: seed_everything(seed)
    
    name = "fashion_mnist_mlp"    
    
    base_path = session_path(name)
    bs = 32
    space_size = 6
    
    
    mnist_dp = FashionMnistDataProvider(batch_size=bs)
    mnist_model = FashionMnistModel(mnist_dp.num_features, 4, mnist_dp.num_classes) #Type: nn.Module
    mnist_space = ModularSpace(name, mnist_model, space_size, useActivations=True, useLayers=True, base_path=base_path)
    
    steps = ModularStep(name, base_path, mnist_dp, mnist_model, mnist_space)
    
    steps.space.rescale_factor = 0.9
    steps.space.observing = True
    steps.space.modularizing = False

    steps.stats.metric_count(False)
    steps.stats.metric_covariance(False)
    
    steps.step_run()


class ModularStep():
    def __init__(self, config: MetaModelConfig, name: str, base_path:str, dp:BaseDataProvider, model:MetaModel, space:ModularSpace):
        self.config = config
        self.name = name
        self.base_path = base_path
        self.train_epochs = config.train_epochs
        self.space_epochs = config.space_epochs

        self.dp = dp
        self.model = model
        self.space = space
        self.model_size = self.model.get_model_size()
        
        
        self.stats = ModularStats(space, base_path = base_path)
        self.vis = ModularSpaceVisualizer(space, base_path = base_path)
        self.autoencoder = config.task == "autoencoder"

        
        self.train_early_stop = 3
        self.train_early_stop_gain = 0.01 # 1% minimum gain
                
        
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "forward_time": [],
            "model_size": [self.model_size],
            "modularity_score": [],
        }

        self.metrics_best = {}
        set_best(self.metrics_best, "model_size", self.model_size)
    


    def step_run(self, train=True, space=False, data=False, cluster=True, purge=True, autoencoder=None, draft=False, save_video=False):
        autoencoder = self.autoencoder if autoencoder is None else autoencoder

        if train:
            self.step_train(epochs=self.train_epochs, autoencoder=autoencoder, draft=draft)
            if save_video: self.vis.play()

        if space: 
            self.step_space(epochs=self.space_epochs)
            if save_video: self.vis.play()
            
        if data:  
            ys, y_hats, avgs, stds = self.step_data()
            self.vis.show_data_per_class(ys, y_hats, avgs, stds)
        
        if cluster:
            self.step_cluster(draft=draft)
            cluster_stats, classes = self.space.cluster_class_stats()
            #modularity_score  = float(cluster_stats['KL_div'].mean())
            modularity_score  = float(cluster_stats['KL_exp'].mean())
            self.metrics['modularity_score'].append( modularity_score )
            set_best(self.metrics_best, "modularity_score", modularity_score)
        
        if purge:
            self.step_ablation(autoencoder=autoencoder)


    def step_train(self, lr=0.001, epochs=20, autoencoder=False, draft=False):  
        print("STEP VALIDATION")
        optim = torch.optim.AdamW(self.model.parameters(), lr=lr)

        options = "+modular" if self.space.modularizing else ""
        
        draft_batches = 20

        early_stop_cnt = 0

        for epoch_num, epoch in tqdm(enumerate(range(epochs)), desc=f"Step TRAIN {options}"):
            train_loss = 0

            train_acc = 0
            train_num = 0
            avg_train_loss = 0
            avg_train_acc = 0
            
            
            batch_bar = tqdm(self.dp.train)
            for x, y in batch_bar:
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                optim.zero_grad()
                y_hat = self.model(x)
                #print('.', end='', flush=True)
                loss = self.model.loss(x, y, y_hat)

                
                loss.backward()
                self.space.step_batch(history=True)

                optim.step()
                
                
                with torch.no_grad():
                    train_loss += loss.detach().item()
                    train_num += len(x)
                    if not autoencoder:
                        y_hat_cls = torch.argmax(y_hat,dim=1)
                        train_acc += torch.sum(y == y_hat_cls)
                
                avg_train_loss = float(train_loss/train_num)
                avg_train_acc =  float(train_acc/train_num)

                desc = f"E {epoch_num+1}/{epochs} | L: {avg_train_loss:.5f} | A: {avg_train_acc:.2f}"
                batch_bar.set_description(desc)
                batch_bar.update()
                if draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft train: Early breaking.')
                        draft_batches = 20
                        break
                

            val_loss, val_acc, forward_time, _ = self.step_validation( compute_cm=False, autoencoder=autoencoder, draft=draft)

            self.metrics["train_loss"].append(avg_train_loss)
            self.metrics["train_acc"].append(avg_train_acc)
            self.metrics["val_loss"].append(val_loss)
            self.metrics["val_acc"].append(val_acc)
            self.metrics["forward_time"].append(forward_time)

            set_best(self.metrics_best, "train_loss", avg_train_loss)
            set_best(self.metrics_best, "train_acc", avg_train_acc, min=False)
            set_best(self.metrics_best, "val_acc", val_acc, min=False)
            val_loss_gain, val_loss_old = set_best(self.metrics_best, "val_loss", val_loss)

            print(  f"[{epoch:4}/{epochs} | Train L:{avg_train_loss:.5f} A:{avg_train_acc:.2f} | Val L:{val_loss:.5f} A:{val_acc:.2f} | T: {forward_time:.2f}s ]")
            self.space.rescale_factor_step()
            self.space.save_space()

            # Early stopping
            gain = val_loss - val_loss_old if val_loss_old is not None else 1
            if self.train_early_stop > 0: # is active
                early_stop_cnt += 1
                if gain >= self.train_early_stop_gain:
                    early_stop_cnt = 0 #valid gain: reset counter
                    print(f'[EarlyStop] New best! +{int(gain*100)}%')
                elif early_stop_cnt > self.train_early_stop:
                    print(f'[EarlyStop] Pain, but gain. Stopping.')
                    break

            
                




            

        self.model.save()

        #vis.play()

    def step_validation(self, compute_cm=True, autoencoder = False, draft=False):
        print("STEP VALIDATION")

        predicted_class = []
        real_class = []

        val_loss = 0
        val_acc = 0
        val_num = 0
        forward_time = time.time()
        
        draft_batches = 20

        with torch.no_grad():
            
            for x, y in self.dp.val: 
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                
                y_hat = self.model(x)
                
                loss = self.model.loss(x, y, y_hat)

                val_loss += loss.detach().item()
                val_num += len(x)

                if self.config.task == "classification":
                    y_hat_cls = torch.argmax(y_hat, dim=1)
                    val_acc += torch.sum(y == y_hat_cls)
                    if compute_cm:
                        predicted_class.extend(y_hat_cls.tolist())
                        real_class.extend(y.tolist())

                if draft:
                    draft_batches -= 1
                    if draft_batches< 0: 
                        print('Draft validation: Early breaking.')
                        draft_batches = 20
                        break

        forward_time = time.time() - forward_time

        cm = None
        if not autoencoder and compute_cm:
            cm = confusion_matrix(real_class, predicted_class)
            #print("Confusion Matrix:\n", cm)

        val_loss = float(val_loss/val_num)
        val_acc = float(val_acc/val_num)
        forward_time = float(forward_time/val_num)
        
        return val_loss, val_acc, forward_time, cm


    def step_space(self, epochs=1, save_history=True):
        print("STEP SPACE")
        #self.model.load()
        self.space.load_space()


        self.model.eval()

        num_batches = len(self.dp.train)
        num = 0 
        for epoch in tqdm(range(epochs) , desc="Step SPACE epoch"):
            for x, y in tqdm(self.dp.train, total=num_batches, desc="Step SPACE batch"):
                x = x.to(device=self.dp.device)
                y = y.to(device=self.dp.device)
                self.model(x)

                #centers_on, centers_off, losses = space.last_centers()
                self.space.step_batch() #centers_on, centers_off, losses)

                if save_history: self.space.add_history()

                num += 1

        self.space.save_space()




    def step_data(self, load=True):
        print("STEP DATA")
        self.dp.shuffle = False
        #self.model.load()
        if load: self.space.load_space()

        self.model.eval()

        self.space.add_history()
        
        #xs = []
        ys = []
        y_hats=[]
        avgs = []
        stds = []
        num_batches = len(self.dp.train)
        for x, y in tqdm(self.dp.train, total=num_batches, desc=f"Step DATA"):
            x = x.to(device=self.dp.device)
            y = y.to(device=self.dp.device)
            y_hat_pred=self.model(x)
            y_hat = torch.argmax(y_hat_pred, dim=1)

            avg, std = self.space.rate()
            #xs.extend(x)
            ys.extend(y.tolist())
            y_hats.extend(y_hat.tolist())
            avgs.extend(avg)
            stds.extend(std)
        
        return ys, y_hats, avgs, stds

    def step_cluster(self, load=False, draft=False):
        print("STEP CLUSTERING")
        self.dp.shuffle = False
        #self.model.load()

        draft_batches = 20
        
        if load: 
            self.space.load_space()
            self.space.load_cluster_map()
            self.space.load_cluster_stats()

        self.model.eval()

        if self.space.last_clustering_all is None or len(self.space.last_clustering_all) == 0:
            self.space.cluster_space()

        if self.space.last_cluster_stats is not None and len(self.space.last_cluster_stats) > 0:
            return
        
        num_batches = len(self.dp.train)
        for x, y in tqdm(self.dp.train, total=num_batches, desc=f"Step CLUSTER-CLASS Attribution"):
            x = x.to(device=self.dp.device)
            y = y.to(device=self.dp.device)
            y_hat_pred = self.model(x)
            y_hat = torch.argmax(y_hat_pred, dim=1)
            self.space.class_cluster_attribution(y, y_hat)

            if draft:
                draft_batches -= 1
                if draft_batches< 0: 
                    draft_batches = 20
                    print('Draft cluster: Early breaking.')
                    break

        self.space.save_cluster_stats()
        


    def step_ablation(self, autoencoder=False, threshold=0.05): #TODO: n_classes, do better
        print("STEP ABLATION")
        self.space.set_pruning_masks()


        acc, valid, _,  cm_base = self.step_validation(compute_cm=True, autoencoder=autoencoder)
        tt_base = truth_table_from_cm(cm_base)
        f1_base = f1s_from_truth_table(tt_base)
        n_classes = len(cm_base)
        print(f"Confusion Matrix:\n{tt_base[:,0]}\n")

        abl = [-1] * len(cm_base)
        cms = [np.column_stack((abl, cm_base))]
        tts = [np.column_stack((abl, tt_base))]
        f1s = [np.concatenate([[-1], f1_base])]
        msk = [[-1]+[len(ls) for ls in self.space.last_clustering]]

        ablated = {}
        for i in range(n_classes):
            masks = self.space.mask_by_class(i, threshold)
            masks_cnt = [ (m == False).sum().item() for m in masks]
            self.space.set_pruning_masks(masks)
            acc, valid, _, cm_class = self.step_validation(compute_cm=True, autoencoder=autoencoder)


            tt_class = truth_table_from_cm(cm_class)
            f1_class = f1s_from_truth_table(tt_class)
            tt_diff = tt_class - tt_base
            f1_diff = f1_base - f1_class

            abl = [i] * len(cm_class)
            cms.append(np.column_stack((abl, cm_class)))
            tts.append(np.column_stack((abl, tt_class)))
            f1s.append(np.concatenate([[i], f1_class]))
            msk.append(np.concatenate([[i], masks_cnt]))

            ablated[i] = {'tt':tt_class, 'f1':f1_class}

            print(f"Class {i} | Delta {tt_diff[i]} | F1 {f1_diff[i]}")
            # print("Confusion Matrix:\n{cm_diff}\n")

        cms = np.concatenate(cms)
        pd.DataFrame.from_records(cms).to_csv(f'{self.base_path}/{self.name}_ablation_cm.csv')

        tts = np.concatenate(tts)
        pd.DataFrame.from_records(tts).to_csv(f'{self.base_path}/{self.name}_ablation_tt.csv')

        f1s = np.array(f1s)
        pd.DataFrame.from_records(f1s).to_csv(f'{self.base_path}/{self.name}_ablation_f1.csv')

        msk = np.array(msk)
        pd.DataFrame.from_records(msk).to_csv(f'{self.base_path}/{self.name}_ablation_msk.csv')

        return cms

if __name__ == "__main__": test()