import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import matplotlib.pyplot as plt
import pandas as pd

class ProTacVisualize():
    def __init__(self, skin_path, init_positions, plot_init = True):
        self.skin_path = skin_path
        self.init_positions = init_positions
        self.positions = np.array(init_positions)
        if plot_init:
            self.plot_initialize()

    def plot_initialize(self):
        self.plotter = pvqt.BackgroundPlotter()
        boring_cmap = plt.cm.get_cmap("bwr")       
        self.skin = pv.read(self.skin_path) # for pyvista visualization
        norm_deviations = np.linalg.norm(self.positions - self.init_positions, axis=1)
        self.skin['contact depth (unit:mm)'] = norm_deviations # for contact depth visualization
        self.skin.points = self.positions
        self.plotter.add_mesh(self.skin, cmap=boring_cmap, clim=[0, 3])