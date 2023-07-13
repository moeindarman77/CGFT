# ---------------------- Contour Comparison ------------------------------- Needs to be checked
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

def comparison_contour_plot(data1, data2):
    """
    Creates two contour plots of 2D matrices side by side for comparison
    
    Args:
        data1 (numpy.ndarray): A 2D array of size M x N for the first plot
        data2 (numpy.ndarray): A 2D array of size M x N for the second plot
    """
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Set font properties
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_size(14)

    # Create contour plots for the two data arrays
    contour_levels = 500
    contour_plot1 = axs[0].contourf(data1, cmap='bwr', levels=contour_levels)
    contour_plot2 = axs[1].contourf(data2, cmap='bwr', levels=contour_levels)
    plt.rcParams["text.usetex"] = True

    # Set nice tick locations for the colorbars
    ticks1 = ticker.MaxNLocator(nbins=5).tick_values(data1.min(), data1.max())
    ticks2 = ticker.MaxNLocator(nbins=5).tick_values(data2.min(), data2.max())

    # Add colorbars to the plots
    colorbar1 = plt.colorbar(contour_plot1, ax=axs[0], ticks=ticks1, shrink = 0.7)
    colorbar2 = plt.colorbar(contour_plot2, ax=axs[1], ticks=ticks2, shrink = 0.7)

    # Set titles and axis labels for the subplots
    axs[0].set_title("Contour Plot 1", fontproperties=font)
    axs[0].set_xlabel("X-axis", fontproperties=font)
    axs[0].set_ylabel("Y-axis", fontproperties=font)
    axs[1].set_title("Contour Plot 2", fontproperties=font)
    axs[1].set_xlabel("X-axis", fontproperties=font)
    axs[1].set_ylabel("Y-axis", fontproperties=font)

    # Set equal aspect ratio for the subplots
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')

    # Set alpha value of the face color to 0
    axs[0].set_facecolor((0, 0, 0, 0))
    axs[1].set_facecolor((0, 0, 0, 0))

    # Show the plot
    plt.show()
