import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def qval_plot(data_sorted_list, score_types):
    """
    Displays the plots of a number of samples from the target database
    with a q-value of no more from following q-value thresholds for each measure
    and saves them to a .png file.
    """
    score_types_number = len(score_types)
    n_pos_examples = []
    qvalues = []
    for i in range(score_types_number):
        n_pos_examples.append(data_sorted_list[i]['pos_neg'].cumsum(axis=0))
        qvalues.append(data_sorted_list[i]['qvalue'])
        plt.plot(qvalues[i], n_pos_examples[i])
    x_lim = 0.2
    y_lim = data_sorted_list[-1][data_sorted_list[-1]['qvalue'] <= x_lim].index.values[-1]
    plt.xlim((0, x_lim))
    plt.ylim((0, y_lim + 1))
    plt.xlabel('q-values')
    plt.ylabel('Number of target samples')
    plt.legend(score_types, loc='lower right')
    plt.title(
        f'Dependence of the number of examples from the target database \n on the q-value threshold for sorting by decreasing score value')
    plt.savefig(f'../plots/Figure_{datetime.now().strftime("%Y-%m-%d %H%M%S")}.png')
    plt.show()


def history_plot(history):
    """
    Plots the model history: accuracy, loss, sensitivity and specificity for training and validation data.
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(12, 9)
    max_epoch = len(history['accuracy'])
    axs[0, 0].plot(history['accuracy'])
    axs[0, 0].plot(history['val_accuracy'])
    axs[0, 0].set(xlim=[0, max_epoch], ylim=[0, 1], xlabel='Epoch number', ylabel='Accuracy',
                  title=' ')
    axs[0, 0].legend(['Training dataset', 'Validation dataset'], loc='lower right')
    axs[0, 0].grid()

    axs[0, 1].plot(history['loss'], 'tab:orange')
    axs[0, 1].plot(history['val_loss'])
    axs[0, 1].set(xlim=[0, max_epoch], ylim=[0, max(max(history['loss'], history['val_loss']))], xlabel='Epoch number',
                  ylabel='Loss', title=' ')
    axs[0, 1].legend(['Training dataset', 'Validation dataset'], loc='upper right')
    axs[0, 1].grid()

    axs[1, 0].plot(history['sensitivity'], 'tab:green')
    axs[1, 0].plot(history['val_sensitivity'])
    axs[1, 0].set(xlim=[0, max_epoch], ylim=[0, 1], xlabel='Epoch number', ylabel='Sensitivity',
                  title=' ')
    axs[1, 0].legend(['Training dataset', 'Validation dataset'], loc='lower right')
    axs[1, 0].grid()

    axs[1, 1].plot(history['specificity'], 'tab:red')
    axs[1, 1].plot(history['val_specificity'])
    axs[1, 1].set(xlim=[0, max_epoch], ylim=[0, 1], xlabel='Epoch number', ylabel='Specificity',
                  title=' ')
    axs[1, 1].legend(['Training dataset', 'Validation dataset'], loc='lower right')
    axs[1, 1].grid()
    plt.savefig(f'../plots/Figure_{datetime.now().strftime("%Y-%m-%d %H%M%S")}.png')
    plt.show()


def histogram_plot(score_types, data):
    """
    For each score type, displays normalized histograms
    of the distribution of the number of samples from the target and decoy database
    with a given value of the score, and saves them to a .png file.
    """
    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(13, 10.5)

    for j, ax in enumerate(fig.axes):
        d = data[data['pos_neg'] == 0][score_types[j]]
        t = data[data['pos_neg'] == 1][score_types[j]]
        counts, edges = np.histogram(t, 50)
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        h = centers[1] - centers[0]
        counts = counts / (sum(counts) * h)
        ax.bar(centers, counts, color='blue', width=h, alpha=0.5)

        counts, edges = np.histogram(d, int((max(d) - min(d)) / h))
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        h = centers[1] - centers[0]
        counts = counts / (sum(counts * h))
        ax.bar(centers, counts, color='red', width=h, alpha=0.5)

        score_name = score_types[j].replace('_', ' ')
        ax.set(xlabel=f"Wartość {score_name}", ylabel='Znormalizowana liczność zbioru', title=' ')
        ax.legend(['Target', 'Decoy'], loc='upper center')
    plt.savefig(f'../plots/Figure_{datetime.now().strftime("%Y-%m-%d %H%M%S")}.png')
    plt.show()
