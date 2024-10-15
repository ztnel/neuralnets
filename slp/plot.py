import mplcursors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_locked_pts(axis, x, y, x_units:str, y_units:str, y_label:str, legend_label: str, style:str = '', color:str = ''):
    """
    Prevent interpolation behaviour in cursors
    """
    if color:
        axis.plot(x, y, style, label=legend_label, color=color)
    else:
        axis.plot(x, y, style, label=legend_label)
    axis.set_ylabel(y_label)
    axis.legend(loc='best')
    # invisible
    dots = axis.scatter(x, y, color='none')
    cursor = mplcursors.cursor(dots, multiple=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"{x[sel.index]} {x_units}\n{round(y[sel.index], 2)} {y_units}"))

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--filename', dest='filename', type=str, help='')
args = parser.parse_args()
df = pd.read_csv( args.filename, sep=r'\s*,\s*', encoding='ascii', engine='python')


error = np.array(list(map(float, df.ERROR)), dtype=float)
bias = np.array(list(map(float, df.BIAS)), dtype=float)
weight_1 = np.array(list(map(float, df.WEIGHT_1)), dtype=float)
weight_2 = np.array(list(map(float, df.WEIGHT_2)), dtype=float)

epoch = df['EPOCH']
samples = len(epoch)

num_signals = len(df.columns) - 1
fig, ax = plt.subplots(nrows=2, sharex=True, squeeze=True)
fig.suptitle(args.filename)
ax[0].minorticks_on()

# set x axis limits
for axis in ax:
    axis.set_xlim((min(epoch), max(epoch)))
# plot power nodes
ax[0].set_title("SLP Parameters")
plot_locked_pts(ax[0], epoch, weight_1, '', '', 'Weight 1', 'w1',color='blue')
plot_locked_pts(ax[0], epoch, weight_2, '', '', 'Weight 2', 'w2', color='blue')
plot_locked_pts(ax[0], epoch, bias, '', '', 'Bias', 'b', color='green')

ax[1].set_title("SLP Error")
plot_locked_pts(ax[1], epoch, error, '', '', 'Error', '', style='r')
fig.supxlabel('Epoch')
plt.show()
