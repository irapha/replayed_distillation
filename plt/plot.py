# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv


def create_plot(x_t, y_t, x_a, y_a, x_label, y_label, plot_name, width, height):
    sns.set(font_scale=2.0)
    sns.set_style({"savefig.dpi": 300})
    sns.set_style("whitegrid")
    sns.set_palette("Set1", 8, .75)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    markersize = 1
    lw=3
    plt.xlim(0, 28100)
    plt.tight_layout()
    # teacher on mnist
    ax.plot([0, x_t[-1]], [0.9891, 0.9891], '#a64d79', markersize=markersize, linewidth=lw)
    # student on mnist #b4a7d6
    ax.plot([0, x_t[-1]], [0.9865, 0.9865], '#d9d2e9', markersize=markersize, linewidth=lw)
    # student distilled.
    ax.plot([0, x_t[-1]], [0.9891, 0.9891], '#a64d79', markersize=markersize, linewidth=lw)
    # top layer
    ax.plot(x_t, y_t, '#c9daf8', markersize=markersize, linewidth=lw)
    # all layers
    ax.plot(x_a, y_a, '#3d85c6', markersize=1, linewidth=lw)
    # all layers + drop
    # ax.plot(x_ad, y_ad, '#6d9eeb', markersize=1, linewidth=lw)
    # spectral all layer
    # ax.plot(x_sa, y_sa, '#674ea7', markersize=1, linewidth=lw)
    # spectral pair layers
    # ax.plot(x_sp, y_sp, '#351c75', markersize=1, linewidth=lw)

    # plt.xlabel(x_label, fontname='Calibri', labelpad=20)
    # plt.ylabel(y_label, fontname='Calibri', labelpad=20)

    grey = '#666666'
    light_grey = '#b7b7b7'
    ax.tick_params(axis='x', colors=grey)
    ax.xaxis.label.set_color(grey)
    ax.tick_params(axis='y', colors=grey)
    ax.yaxis.label.set_color(grey)

    # fig.savefig(plot_name)
    # plt.close()
    # plt.show()
    plt.savefig(plot_name)

with open('convtoplayer.csv', 'rb') as csvfile:
    d_loss = csv.reader(csvfile, delimiter=',')
    data = [(row[1], row[2]) for row in d_loss]
    x_t, y_t = zip(*data[1:])

with open('convalllayer.csv', 'rb') as csvfile_2:
    d_loss_2 = csv.reader(csvfile_2, delimiter=',')
    data_2 = [(row[1], row[2]) for row in d_loss_2]
    x_a, y_a = zip(*data_2[1:])

# with open('alllayerdrop.csv', 'rb') as csvfile_2:
    # d_loss_2 = csv.reader(csvfile_2, delimiter=',')
    # data_2 = [(row[1], row[2]) for row in d_loss_2]
    # x_ad, y_ad = zip(*data_2[1:])

# with open('spectralalllayer.csv', 'rb') as csvfile_2:
    # d_loss_2 = csv.reader(csvfile_2, delimiter=',')
    # data_2 = [(row[1], row[2]) for row in d_loss_2]
    # x_sa, y_sa = zip(*data_2[1:])

# with open('spectralpairlayer.csv', 'rb') as csvfile_2:
    # d_loss_2 = csv.reader(csvfile_2, delimiter=',')
    # data_2 = [(row[1], row[2]) for row in d_loss_2]
    # x_sp, y_sp = zip(*data_2[1:])

create_plot(x_t, y_t, x_a, y_a, 'training iteration', 'test set accuracy', 'fully_connected', 10, 5.5)

