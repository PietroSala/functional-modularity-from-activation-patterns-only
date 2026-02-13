import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

import torch

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


### 
from modularnet.modularnet.modularspace import ModularSpace


class StrangeSpaceVisualizerPCA:
    def __init__(self,  space: ModularSpace):
        matplotlib.use('TkAgg')
        self.space = space
        self.position_history = self.space.position_history
        self.pca = None
        self.fitted = False

    def fit_pca(self):
        # Combine all embeddings from all timesteps
        all_embeddings = []
        for timestep in self.position_history:
            for group in timestep:
                if isinstance(group, torch.Tensor):
                    group = group.detach().cpu().numpy()
                all_embeddings.append(group)
        
        all_embeddings = np.vstack(all_embeddings)

        # Fit PCA on all embeddings
        self.pca = PCA(n_components=2)
        self.pca.fit(all_embeddings)
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

    def transform_all(self):
        if not self.fitted:
            self.fit_pca()

        transformed_history = []
        for timestep in self.position_history:
            transformed_timestep = self.transform_timestep(timestep)
            transformed_history.append(transformed_timestep)
        
        return transformed_history

    def play(self, interval=1, figsize=(10, 10)):
        if not self.fitted: self.fit_pca()

        transformed_history = self.transform_all()

        # Calculate the overall min and max for x and y
        all_data = np.vstack([np.vstack(timestep) for timestep in transformed_history])
        x_min, y_min = np.min(all_data, axis=0)
        x_max, y_max = np.max(all_data, axis=0)

        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        fig, ax = plt.subplots(figsize=figsize)
        
        # Get a color for each group
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        scatter_plots = []
        for i, group in enumerate(transformed_history[0]):
            color = colors[i % len(colors)]
            scatter = ax.scatter([], [], c=color, label=f'Layer {i+1}', alpha=0.2, s=100)
            scatter_plots.append(scatter)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.legend()
        ax.set_title('Embedding Visualization')
        
        def update(frame):
            #ax.clear()
            for i, scatter in enumerate(scatter_plots):
                data = transformed_history[frame][i]
                scatter.set_offsets(data)
            
            ax.set_title(f'{frame+1}')
            #ax.set_title(f'Embedding Visualization - Timestep {frame+1}')
            return scatter_plots

        anim = FuncAnimation(fig, update, frames=len(transformed_history), 
                             interval=interval, blit=False)
        
        #plt.show()
        plt.show(block=True)
        return anim


class StrangeSpaceVisualizer2D:
    def __init__(self, space: ModularSpace):
        matplotlib.use('TkAgg')
        self.space = space
        self.position_history = self.space.position_history
        self.original_xlim = None
        self.original_ylim = None
        self.anim = None
        self.background = None
        self.colors = list(mcolors.TABLEAU_COLORS.values())

    def show_data(self, ys, y_hats, avgs, stds, figsize=(10,10)):
        fig, ax = plt.subplots(figsize=figsize)

        #avgs = avgs.numpy()
        #stds = stds.numpy()

        positions_layers = self.space.positions
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
        plt.show(block=True)


    def play(self, interval=0.001, figsize=(10, 10)):
        history = self.position_history
        all_data = np.vstack([np.vstack(timestep) for timestep in history])
        x_min, y_min = np.min(all_data, axis=0)
        x_max, y_max = np.max(all_data, axis=0)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min, x_max = x_min - x_padding, x_max + x_padding
        y_min, y_max = y_min - y_padding, y_max + y_padding

        self.fig, self.ax = plt.subplots(figsize=figsize)
        #plt.ion()
        #self.ax.set_aspect('equal')
        self.scatter_plots = []
        for i, group in enumerate(history[0]):
            color = self.colors[i % len(self.colors)]
            scatter = self.ax.scatter([], [], c=color, label=f'Layer {i+1}', alpha=0.2, s=100)
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

        self.anim = FuncAnimation(self.fig, update, frames=len(history),
                                  interval=interval, blit=True)

        def zoom(event):
            base_scale = 1.1
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            cur_xrange = (cur_xlim[1] - cur_xlim[0]) * 0.5
            cur_yrange = (cur_ylim[1] - cur_ylim[0]) * 0.5
            xdata = event.xdata
            ydata = event.ydata

            if event.button == 'up':
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                scale_factor = base_scale
            else:
                scale_factor = 1

            new_xlim = [xdata - cur_xrange * scale_factor,
                        xdata + cur_xrange * scale_factor]
            new_ylim = [ydata - cur_yrange * scale_factor,
                        ydata + cur_yrange * scale_factor]

            
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        def reset(event):
            if event.key == ' ':  # Spacebar
                self.ax.set_xlim(self.original_xlim)
                self.ax.set_ylim(self.original_ylim)
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

        def on_press(event):
            if event.button != 1: return # Only respond to left mouse button
            self._pan_start = (event.xdata, event.ydata)
            self.anim.pause()  # Pause the animation
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

        def on_release(event):
            if event.button != 1: return  # Only respond to left mouse button
            self._pan_start = None
            self.anim.resume()  # Resume the animation
            self.background = None

        def on_motion(event):
            if self._pan_start is None: return
            dx = event.xdata - self._pan_start[0]
            dy = event.ydata - self._pan_start[1]
            cur_xlim = self.ax.get_xlim()
            cur_ylim = self.ax.get_ylim()
            self.ax.set_xlim(cur_xlim - dx)
            self.ax.set_ylim(cur_ylim - dy)
            self._pan_start = (event.xdata, event.ydata)

            if self.background is not None:
                self.fig.canvas.restore_region(self.background)
                for scatter in self.scatter_plots:
                    self.ax.draw_artist(scatter)
                self.fig.canvas.blit(self.ax.bbox)
                self.fig.canvas.flush_events()
                self.fig.canvas.draw_idle()  # Ensure the background is properly cleared

        self._pan_start = None
        self.fig.canvas.mpl_connect('scroll_event', zoom)
        self.fig.canvas.mpl_connect('key_press_event', reset)
        self.fig.canvas.mpl_connect('button_press_event', on_press)
        self.fig.canvas.mpl_connect('button_release_event', on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', on_motion)

        plt.show()
        return self.anim