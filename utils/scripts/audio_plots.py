import matplotlib.pyplot as plt

def pretty_plot(signal, title, path):
    fig,ax = plt.subplots()
    plt.plot(signal)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title(title)
    plt.savefig(path)
    plt.clf()

def plot_signals(signal1, signal2, labelx, labely, path, c):
    fig,ax = plt.subplots()
    plt.scatter(signal1, signal2)
    plt.title(c)
    plt.xticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel(labelx, fontsize=15)
    plt.ylabel(labely, fontsize=15)
    plt.savefig(path)
    plt.clf()

