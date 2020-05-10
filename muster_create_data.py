import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(1234)
import pickle


# Make signal and background samples with up and down shifts in background process
def make_sig(num_events):
    mean = [-1, -1]
    corr = 0.0
    cov = [[1.0, corr], [corr, 1.0]]
    return np.random.multivariate_normal(mean, cov, num_events)


def make_bkg(num_events, shift):
    mean = [1, 1 + shift]
    corr = 0.0
    cov = [[1.0, corr], [corr, 1.0]]
    return np.random.multivariate_normal(mean, cov, num_events)


def make_dataset(num_events, shift):
    num_events = num_events // 2
    x = np.vstack([make_sig(num_events), make_bkg(num_events, shift)])
    y = np.hstack([np.ones(num_events), np.zeros(num_events)])
    return x, y


# Create training dataset
num_train = 100000
total_shift = 1.0
signal_exp = 1000
background_exp = 1000
signal_scale = signal_exp / float(num_train)
background_scale = background_exp / float(num_train)

x_train, y_train = make_dataset(num_train, 0)
x_train_up, _ = make_dataset(num_train, total_shift)
x_train_down, _ = make_dataset(num_train, -total_shift)

w_train = np.ones(y_train.shape)
w_train[y_train == 1] = signal_scale
w_train[y_train == 0] = background_scale


# Create testing dataset
num_test = num_train # Don't change this!

x_test, y_test = make_dataset(num_test, 0)
x_test_up, _ = make_dataset(num_test, total_shift)
x_test_down, _ = make_dataset(num_test, -total_shift)

w_test = np.ones(y_train.shape)
w_test[y_test == 1] = signal_scale
w_test[y_test == 0] = background_scale

# Plot data as contour plot
plt.figure(figsize=(6, 6))
bins = (20, 20)
xlim = (-4, 4)
ylim = (-4, 4)
h_sig, xe, ye = np.histogram2d(x_test[y_test == 1][:, 0], x_test[y_test == 1][:, 1],
        bins=bins, range=(xlim, ylim), density=True)
h_bkg, _, _ = np.histogram2d(x_test[y_test == 0][:, 0], x_test[y_test == 0][:, 1],
        bins=bins, range=(xlim, ylim), density=True)
h_bkg_up, _, _ = np.histogram2d(x_test_up[y_test == 0][:, 0], x_test_up[y_test == 0][:, 1],
        bins=bins, range=(xlim, ylim), density=True)
h_bkg_down, _, _ = np.histogram2d(x_test_down[y_test == 0][:, 0], x_test_down[y_test == 0][:, 1],
        bins=bins, range=(xlim, ylim), density=True)
norm=plt.Normalize(0,1)
cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0"] * 3)
cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C1"] * 3)
plt.contour(h_bkg.T, extent=[xe[0], xe[-1], ye[0], ye[-1]], linewidths=3, cmap=cmap2, alpha=0.8)
plt.contour(h_bkg_up.T, extent=[xe[0], xe[-1], ye[0], ye[-1]], linewidths=3, cmap=cmap2, alpha=0.4)
plt.contour(h_bkg_down.T, extent=[xe[0], xe[-1], ye[0], ye[-1]], linewidths=3, cmap=cmap2, alpha=0.4)
plt.contour(h_sig.T, extent=[xe[0], xe[-1], ye[0], ye[-1]], linewidths=3, cmap=cmap1, alpha=0.8)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(xlim)
plt.ylim(ylim)
#plt.savefig("data.png", bbox_inches="tight")
plt.show()


# Write dataset to file
#pickle.dump([x_train, x_train_up, x_train_down, y_train, w_train],
#            open("train.pickle", "wb"))
#pickle.dump([x_test, x_test_up, x_test_down, y_test, w_test],
#            open("test.pickle", "wb"))