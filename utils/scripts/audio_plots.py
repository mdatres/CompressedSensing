import matplotlib.pyplot as plt

def pretty_plot(signal, title, path):
    plt.plot(signal)
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def plot_signals(signal1, signal2, labelx, labely, path):
    plt.plot(signal1, signal2)
    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.savefig(path)
    plt.clf()

