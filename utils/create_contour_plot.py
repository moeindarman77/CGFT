# ---------------------- Contour Plot -------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker

def create_contour_plot(data):
    """
    Creates a contour plot of a 2D matrix
    
    Args:
        data (numpy.ndarray): A 2D array of size M x N
    """
    # Create a contour plot with a blue-white-red colorbar
    fig, ax = plt.subplots()

    # Set font properties
    font = FontProperties()
    font.set_family('Times New Roman')
    font.set_size(14)

    # Create a contour plot
    contour_levels = 500
    contour_plot = ax.contourf(data, cmap='bwr', levels=contour_levels)
    plt.rcParams["text.usetex"] = True

    # Set nice tick locations for the colorbar
    ticks = ticker.MaxNLocator(nbins=5).tick_values(data.min(), data.max())

    # Add a colorbar to the plot
    colorbar = plt.colorbar(contour_plot, ticks=ticks)

    # Set title and axis labels
    ax.set_title("Contour Plot", fontproperties=font)
    ax.set_xlabel("X-axis", fontproperties=font)
    ax.set_ylabel("Y-axis", fontproperties=font)
    ax.set_facecolor((0, 0, 0, 0))  # Set alpha value of the face color to 0

    # Show the plot
    plt.show()
