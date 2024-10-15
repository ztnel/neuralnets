import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

epoch_buf = []
err_buf = []
w1_buf = []
w2_buf = []
bias_buf = []

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title("SLP Parameters")
ax2.set_title("SLP Error")
fig.suptitle('Single Layer Perceptron Training')
fig.supxlabel('Training Epoch')

# Initialize two line objects (one in each axes)
l1, = ax1.plot([], [], label="w1", color='blue')
l2, = ax1.plot([], [], label="w2", color='green')
l3, = ax1.plot([], [], label="bias", color='orange')
l4, = ax2.plot([], [], label="error", color='red')
line = [l1, l2, l3, l4]

# Set axis labels
ax1.set_ylabel('Weights/Bias')
ax2.set_ylabel('Error')
ax2.set_xlabel('Epoch')

# Add legends to the plots
ax1.legend()
ax2.legend()

# Initialize the limits for the axes
ax1.set_xlim(0, 10)
ax1.set_ylim(-1, 1)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 1)

def init():
    """
    Initialization function for FuncAnimation.
    Sets the line objects with empty data.
    """
    for l in line:
        l.set_data([], [])
    return line

def read_data():
    """
    Reads data from stdin and yields it to be used in the animation.
    """
    while True:
        line = sys.stdin.readline()
        if line:
            # Split the line and convert to appropriate types
            epoch, w1, w2, bias, err = line.strip().split(',')
            yield int(epoch), float(w1), float(w2), float(bias), float(err)

def run(data):
    """
    Redraw the plot with the latest data.
    """
    epoch, w1, w2, bias, err = data
    epoch_buf.append(epoch)
    w1_buf.append(w1)
    w2_buf.append(w2)
    bias_buf.append(bias)
    err_buf.append(err)
    
    # Dynamically adjust axis limits if needed
    xmin, xmax = ax1.get_xlim()
    if epoch >= xmax:
        ax1.set_xlim(xmin, 2 * xmax)
        ax2.set_xlim(xmin, 2 * xmax)
        ax1.figure.canvas.draw()
        ax2.figure.canvas.draw()
    
    ymin1, ymax1 = ax1.get_ylim()
    ymin2, ymax2 = ax2.get_ylim()
    
    if min(w1_buf + w2_buf + bias_buf) <= ymin1 or max(w1_buf + w2_buf + bias_buf) >= ymax1:
        ax1.set_ylim(min(w1_buf + w2_buf + bias_buf) - 0.1, max(w1_buf + w2_buf + bias_buf) + 0.1)
    
    if min(err_buf) <= ymin2 or max(err_buf) >= ymax2:
        ax2.set_ylim(min(err_buf) - 0.1, max(err_buf) + 0.1)

    # Update the data for each line
    line[0].set_data(epoch_buf, w1_buf)
    line[1].set_data(epoch_buf, w2_buf)
    line[2].set_data(epoch_buf, bias_buf)
    line[3].set_data(epoch_buf, err_buf)

    return line

if __name__ == '__main__':
    # Create the animation
    ani = animation.FuncAnimation(fig, run, read_data, init_func=init, interval=1, blit=True, repeat=False, cache_frame_data=False)
    plt.show()
