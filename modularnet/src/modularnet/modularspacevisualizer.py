import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import pandas as pd

import matplotlib.cm as cm
import torch

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


### 
from modularnet.modularspace import ModularSpace
from scipy.stats import entropy


class ModularSpaceVisualizer:
    def __init__(self, space: ModularSpace, base_path=None):
        #matplotlib.use('TkAgg') # UI
        matplotlib.use('Agg')
        self.space = space
        self.base_path = base_path
        self.position_history = self.space.position_history
        self.original_xlim = None
        self.original_ylim = None
        self.anim = None
        self.background = None
        self.colors = list(mcolors.TABLEAU_COLORS.values())
        self.pca = None
        self.fitted = False
        self.is_playing = False

    def fit_pca(self, data=None):
        if data is None: data = self.position_history
        # Combine all embeddings from all timesteps
        all_embeddings = []
        for timestep in data:
            for group in timestep:
                if isinstance(group, torch.Tensor):
                    group = group.detach().cpu().numpy()
                all_embeddings.append(group)
        
        all_embeddings = np.vstack(all_embeddings)

        # Fit PCA on all embeddings
        self.pca = PCA(n_components=2)
        self.pca.fit(all_embeddings)

        cs = np.cumsum(self.pca.explained_variance_ratio_)
        print("PCA variance: ", cs)
        self.fitted = True

    def transform_timestep(self, timestep):
        if not self.fitted:
            raise ValueError("PCA not fitted. Call fit_pca() first.")

        transformed_timestep = []
        for group in timestep:
            if isinstance(group, torch.Tensor):
                group = group.detach().cpu().numpy()
            transformed_group = self.pca.transform(group)
            transformed_timestep.append(transformed_group)
        
        return transformed_timestep

    def transform_all(self, data=None):
        if data is None: data = self.position_history

        if not self.fitted:
            self.fit_pca(data)

        transformed_history = []
        for timestep in data:
            transformed_timestep = self.transform_timestep(timestep)
            transformed_history.append(transformed_timestep)
        
        return transformed_history

    def show_cluster_class_attribution(self):
        cluster_stats, classes = self.space.cluster_class_stats()

        col_totals = cluster_stats[classes].sum(axis=1)
        # Sort clusters by size
        clusters_sum = {k: v for k, v in sorted(clusters_sum.items(), key=lambda item: sum(item[1].values()), reverse=True)}
        # Sort classes by size
        for cluster_num, class_sum in clusters_sum.items():
            clusters_sum[cluster_num] = {k: v for k, v in sorted(class_sum.items(), key=lambda item: item[1], reverse=True)}
        # Plot clusters and classes
        


        self.pie_chart_grid(clusters_sum)
        return
        # each cluster contains the number hits, the times a neuron belonging to that cluster activated for a certain clas
        # for each cluster, plot a pie chart
        # each pie chart is labeled with the cluster number
        # each pie chat is subdivided into the classes composing it, proportionally.
        # each pie chart is colored according to the class
        # each pie chart is labeled with the class and the percentage of hits for that cluster.
        # each pie chart overall size is proportinal to the total number of hits for that cluster
        # each piechat has a minimum and maximum size that allow for a good visualization 
        # each pie chart is placed in x,y coordinates that are not overlapping with other pie charts 
        # the pie chart coordinates are taken from self.space.last_cluster_centers after being reduced with PCA to 2D       

        
        # Save the plot
        plt.savefig(f"{self.base_path}/cluster_class_attribution.png", dpi=300, bbox_inches='tight')
        plt.show(block=True)
        plt.close(fig)

       
    

    def pie_chart_grid(self, clusters_sum):
        import matplotlib.cm as cm

        fig, axs = plt.subplots(
            nrows=math.ceil(len(clusters_sum) ** 0.5),
            ncols=math.ceil(len(clusters_sum) ** 0.5),
            figsize=(20, 20)
        )
        axs = axs.flatten()

        # Find all unique classes for color mapping
        all_classes = set()
        for class_sum in clusters_sum.values():
            all_classes.update(class_sum.keys())
        all_classes = sorted(all_classes)
        class_to_color = {c: cm.tab20(i % 20) for i, c in enumerate(all_classes)}

        # Determine min/max pie size
        sizes = [sum(class_sum.values()) for class_sum in clusters_sum.values()]
        min_size, max_size = 100, 500
        min_cnt, max_cnt = min(sizes), max(sizes)
        def scale_size(cnt):
            if max_cnt == min_cnt:
                return (min_size + max_size) / 2
            return min_size + (cnt - min_cnt) / (max_cnt - min_cnt) * (max_size - min_size)

        for ax in axs[len(clusters_sum):]:
            ax.axis('off')

        for idx, (cluster_num, class_sum) in enumerate(clusters_sum.items()):
            ax = axs[idx]
            total = sum(class_sum.values())
            labels = []
            fracs = []
            colors = []
            autopct = []
            for c, cnt in class_sum.items():
                labels.append(f"{c}")
                fracs.append(cnt)
                colors.append(class_to_color[c])
                autopct.append(f"{cnt/total*100:.1f}%")
            # Pie chart
            ax_pie = ax.pie(
                fracs,
                labels=labels,
                colors=colors,
                startangle=90,
                radius=1.0
            )
            wedges, texts = ax_pie
            # Add class labels and percentages
            label_dist_class = 0.8
            label_dist_perc = 1.2
            for i, (w, c, cnt) in enumerate(zip(wedges, labels, fracs)):
                ang = (w.theta2 + w.theta1) / 2
                x = math.cos(math.radians(ang))
                y = math.sin(math.radians(ang))
                ax.text(
                    label_dist_class * x, label_dist_class * y,
                    f"{c}",
                    ha='center', va='center', fontsize=20
                )
                ax.text(
                    label_dist_perc * x, label_dist_perc * y,
                    f"{cnt/total*100:.1f}%",
                    ha='center', va='center', fontsize=10
                )
            # Set pie size
            pie_size = scale_size(total)
            for w in wedges:
                w.set_radius(pie_size / max_size)
            ax.set_title(f"Cluster {cluster_num} - Total: {total}", fontsize=12)
            ax.axis('equal')
            ax.axis('off')

        # Add legend for classes
        handles = [matplotlib.patches.Patch(color=class_to_color[c], label=f"Class {c}") for c in all_classes]
        fig.legend(handles=handles, loc='upper right' ) #, bbox_to_anchor=(1.15, 1))
        fig.suptitle("Cluster-Class Attribution", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])


        plt.savefig(f"{self.base_path}/cluster_class_attribution.png", dpi=300, bbox_inches='tight')
        plt.show(block=True)
        plt.close(fig)




    def show_data_per_class(self, ys, y_hats, avgs, stds, figsize=(20,20)):
        #fig, ax = plt.subplots(figsize=figsize)

        #avgs = avgs.numpy()
        #stds = stds.numpy()

        positions_layers = self.space.positions
        if self.space.space_size > 2:
            positions_layers = self.transform_all([positions_layers])[0]
            avgs = self.pca.transform(avgs)
            stds = self.pca.transform(stds)

        data = {}
        for y, y_hat, avg, std in zip(ys, y_hats, avgs, stds):
            #key = ( y, tuple(y_hat), tuple(avg), tuple(std))
            #key = ( y, y_hat) #, tuple(avg), tuple(std))
            

            key = y if y_hat == y else -1
            if key not in data: data[key] = [0,[],[],[]]
            data[key][0] += 1
            data[key][1].append(y_hat)
            data[key][2].append(avg)
            data[key][3].append(std)
        
        keys = list(data.keys())
        keys.sort()
        cnt_plots = len(keys)

        rows = max(math.ceil(math.sqrt(cnt_plots)),1)
        cols = max(math.ceil(cnt_plots/rows),1)

        for plot_n, key in enumerate( keys ):
            data_cls = data[key]
            cnt, y_hats, avgs, stds = data_cls

            print(f'Plotting {plot_n+1}/{cnt_plots}')

            ax = plt.subplot( rows, cols, plot_n+1)
            ax.set_title(f"Class {key}")
            scatters_layer = []
            for i, (pos, color) in enumerate(zip(positions_layers, self.colors)):
                p = np.array(pos)
                scatter =  ax.scatter(p[:,0], p[:,1], c=color, s=100, label=f'Layer {i}', alpha=0.1  )
                scatters_layer.append(scatter)

            for c,xy in enumerate(positions_layers[-1]):
                tx,ty = np.array(xy)
                ax.text(tx,ty, s=f'REAL {c}', color=color)
            

            color = self.colors[y]
            av = np.array(avgs)
            st = np.array(stds)
            marker = 'x'  #if y != y_hat else 'v'
            scatter =  ax.scatter(av[:,0], av[:,1], c=color, s=100, label=f'Pred {y_hat}, Cnt: {cnt}', marker=marker )
            
            """
            confidence = 0.95
            n_std = np.sqrt(2) * np.sqrt(-2 * np.log(1 - confidence))
            
            for (tx,ty),(sx,sy) in zip(av,st):
                rx = tx #+ np.random.uniform(-0.1,0.1)
                ry = ty #+ np.random.uniform(-0.1,0.1)
                ax.text(rx,ry, s=f'Class {y} - {cnt}', color=color, alpha=0.1)
                ellipse = Ellipse((tx, ty), width=sx*n_std*2, height=sy*n_std*2, facecolor='none', edgecolor=color, alpha=0.2)
                ax.add_patch(ellipse)
            """
        plt.legend()
        plt.savefig(f"{self.base_path}/data_per_class.png", dpi=300)

        plt.show(block=False)
        
    def show_data(self, ys, y_hats, avgs, stds, figsize=(10,10)):
        fig, ax = plt.subplots(figsize=figsize)

        #avgs = avgs.numpy()
        #stds = stds.numpy()
        ax.set_title(f"Class {y}")

        positions_layers = self.space.positions
        if self.space.space_size > 2:
            positions_layers = self.transform_all([positions_layers])

        scatters_layer = []
        for i, (pos, color) in enumerate(zip(positions_layers, self.colors)):
            p = pos.detach().cpu().numpy()
            scatter =  ax.scatter(p[:,0], p[:,1], c=color, s=100, label=f'Layer {i}', alpha=0.1  )
            scatters_layer.append(scatter)

        for c,xy in enumerate(positions_layers[-1]):
            tx,ty = xy.detach().cpu().numpy()
            ax.text(tx,ty, s=f'REAL {c}', color=color)
        
                        
        data = {}
        for y, y_hat, avg, std in zip(ys, y_hats, avgs, stds):
            #key = ( y, tuple(y_hat), tuple(avg), tuple(std))
            key = ( y, y_hat) #, tuple(avg), tuple(std))
            if key not in data: data[key] = [0,[],[]]
            data[key][0] += 1
            data[key][1].append(avg)
            data[key][2].append(std)

        for (y, y_hat), (cnt, avgs, stds) in data.items():
            color = self.colors[y]
            av = np.array(avgs)
            st = np.array(stds)
            marker = 'x' if y != y_hat else 'v'
            scatter =  ax.scatter(av[:,0], av[:,1], c=color, s=100, label=f'Class {y}, Pred {y_hat}, Cnt: {cnt}', marker=marker )
            

            confidence = 0.95
            n_std = np.sqrt(2) * np.sqrt(-2 * np.log(1 - confidence))

            for (tx,ty),(sx,sy) in zip(av,st):
                rx = tx #+ np.random.uniform(-0.1,0.1)
                ry = ty #+ np.random.uniform(-0.1,0.1)
                ax.text(rx,ry, s=f'Class {y} - {cnt}', color=color, alpha=0.1)
                ellipse = Ellipse((tx, ty), width=sx*n_std*2, height=sy*n_std*2, facecolor='none', edgecolor=color, alpha=0.2)
                ax.add_patch(ellipse)

        plt.legend()
        plt.show(block=False)
        plt.savefig(f"{self.base_path}/show_data.png", dpi=300)


    def play(self, data=None, figsize=(10, 10), interval=1, save=True, block=True, show=False):
        if data is None: data = self.position_history

        if self.space.space_size > 2:
            self.fit_pca(data)
            history = self.transform_all(data)
        else:
            history = data

        all_data = np.vstack([np.vstack(timestep) for timestep in history])
        x_min, y_min = np.min(all_data, axis=0)
        x_max, y_max = np.max(all_data, axis=0)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        self.fig, self.ax = plt.subplots(figsize=figsize)

        #plt.ion()
        self.ax.set_aspect('equal')


        self.scatter_plots = []
        for i, group in enumerate(history[0]):
            color = self.colors[i % len(self.colors)]
            alpha = min(1.0,(max(10/len(group),0.05)))
            scatter = self.ax.scatter([], [], c=color, label=f'Layer {i+1}', alpha=alpha, s=100)
            self.scatter_plots.append(scatter)

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.original_xlim = self.ax.get_xlim()
        self.original_ylim = self.ax.get_ylim()
        self.ax.legend()
        self.ax.set_title('Embedding Visualization')

        def update(frame):
            for i, scatter in enumerate(self.scatter_plots):
                data = history[frame][i]
                scatter.set_offsets(data)
            
            self.ax.set_title(f'{frame+1}')
            return self.scatter_plots

        self.anim = FuncAnimation(self.fig, update, frames=len(history), interval=interval)
                
        self.fig.canvas.mpl_connect('key_press_event', self.play_key_press)
        self.is_playing = True
        if save:
            print("STEP SPACE: saving video...")
            self.anim.save(f'{self.base_path}step_space_{self.space.name}.mp4', fps=30)
        
        if show:
            plt.show(block=block)
        
        #return self.anim
    
    def play_key_press(self,event):
        print("key pressed: ", event.key)
        key = event.key.lower()
        if key == 'q':  
            self.ax.set_xlim(self.original_xlim)
            self.ax.set_ylim(self.original_ylim)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        elif key == ' ': # Spacebar
            if self.is_playing:
                self.anim.pause()
            else:
                self.anim.resume()
            self.is_playing = not self.is_playing
            