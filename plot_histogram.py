import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.random.set_seed(1234)   
import tensorflow_probability as tfp
import pickle
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


####
#### Plot histogram displaying significance
####

## Load data
s, b, b_up, b_down, bins = pickle.load(open("plot_histogram.pickle", "rb"))
plot_label = pickle.load(open("plot_label.pickle", "rb"))

## Scale data
if(plot_label == "_SD_with_nuisance"):
    s = 3 * s
    b = b + b_up + b_down


n = len(s)
bins_for_plots_middle = []                  # Central Point of Bin
for i in range(0, len(bins) - 1):
    bins_for_plots_middle.append(bins[i] + (bins[i + 1] - bins[i]) / 2)
border = 0.5

plt.figure(figsize=(7, 6))
plt.hist(bins_for_plots_middle, weights= [s[0], s[1]], bins= bins, histtype="step", label="Signal", lw=2)
plt.hist(bins_for_plots_middle, weights= [b[0], b[1]], bins= bins, histtype="step", label="Backgorund", lw=2)
plt.legend(loc= "lower center")
plt.xlabel("Projection with decision boundary from NN at {}".format(border))
plt.ylabel("# Events")
plt.axvline(x=border, ymin=0, ymax=n, color="r", linestyle= "dashed", lw=2)
plt.savefig("./plots/histogram_{}.png".format(plot_label), bbox_inches = "tight")
