"""
Interactive plotter to draw circles on clusters, saving cluster radii.
"""

from astropy.io import ascii
import h5py
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import random

stddev = 0.017359

class CircleDrawer:
    def __init__(self, ax):
        self.ax = ax
        self.cutouts = h5py.File('../processed/cutouts.hdf')
        self.table = ascii.read('../processed/camira_final.tbl',  
                                names = ["ID", 
                                         "Name", 
                                         "RA [deg]", 
                                         "Dec [deg]", 
                                         "z_cl", 
                                         "Richness", 
                                         "BCG redshift"]
                    )
        self.possible_ids = [str(x) for x in range(len(self.table))]
        self.result = {}
        self.curr_id = random.choice(self.possible_ids)
        self.possible_ids.pop(int(self.curr_id))

        # Display first cluster
        self.ax.imshow(self.stretch(self.cutouts[self.curr_id]['HDU0']['DATA']), origin='lower')
        self.ax.set_title(self.curr_id)
        plt.draw()

    
    def stretch(self, cutout):
        return np.arcsinh(np.clip(cutout, a_min=0.0, a_max=10.0) / stddev)
        
    # Draw a random next cluster
    def next(self, event):
        next_id = random.choice(self.possible_ids)
        self.curr_id = next_id
        self.possible_ids.pop(int(self.curr_id)) # Remove from possible choices
        self.ax.clear()
        self.ax.imshow(self.stretch(self.cutouts[self.curr_id]['HDU0']['DATA']), origin='lower')
        self.ax.set_title(self.curr_id)
        plt.draw()

    def on_click(self, event):
        if event.inaxes in [self.ax]:
            self.draw_circle(event)
    
    # Draw a circle on click
    def draw_circle(self, event):
        if len(self.ax.patches) > 0:
            self.ax.patches[0].remove()
        x, y = event.xdata, event.ydata
        cenx, ceny = self.ax.get_xlim()[1] // 2, self.ax.get_ylim()[1] // 2
        radius = np.sqrt((x-cenx)**2 + (y-ceny)**2)
        circle = Circle((cenx, ceny), radius, fill=False, color='r')
        self.ax.add_patch(circle)
        # Add to results
        self.result[self.curr_id] = radius
        plt.draw()
    
    # Clear any circles from axis and result dictionary
    def clear(self, event):
        if len(self.ax.patches) > 0:
            self.ax.patches[0].remove()
        if self.curr_id in self.result:
            self.result.pop(self.curr_id)
        plt.draw()
    
    # Close the plot
    def close(self, event):
        plt.close()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    drawer = CircleDrawer(ax)
    axclear = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    axclose = fig.add_axes([0.59, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(drawer.next)
    bclear = Button(axclear, 'Clear')
    bclear.on_clicked(drawer.clear)
    bclose = Button(axclose, 'Close')
    bclose.on_clicked(drawer.close)

    fig.canvas.mpl_connect('button_press_event', drawer.on_click)

    plt.show()

    # After plot is closed
    print(drawer.result)
    