import numpy as np
import pandas as pd

# Hack to keep matplotlib from crashing when run without X
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Apply sane defaults to matplotlib
import seaborn as sns
sns.set_style('darkgrid')


def plot_xy(x, y, x_axis="X", y_axis="Y", title="Plot"):
    df = pd.DataFrame({'x': x, 'y': y})
    plot = df.plot(x='x', y='y')

    plot.grid(b=True, which='major')
    plot.grid(b=True, which='minor')
    
    plot.set_title(title)
    plot.set_ylabel(y_axis)
    plot.set_xlabel(x_axis)
    return plot
