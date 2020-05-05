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
    mean = np.array([1, 1])
    mean = mean + shift
    corr = 0.0
    cov = [[1.0, corr], [corr, 1.0]]
    return np.random.multivariate_normal(mean, cov, num_events)


def make_dataset(num_events, shift):
    num_events = num_events // 2
    x = np.vstack(make_sig(num_events))
    y = np.vstack(make_bkg(num_events, shift))
    return x, y


# Create training dataset
num_train = 100000
total_shift = 1.0
signal_exp = 1000
background_exp = 1000
shift = np.array([0.0, 1.0])


# dataset with events containing each x-/y-coordinates
x_train_noshift_signal, x_train_noshift_background = make_dataset(num_train, 0)
x_train_upshift_signal, x_train_upshift_background = make_dataset(num_train, shift)
x_train_downshift_signal, x_train_downshift_background = make_dataset(num_train, -shift)
x_train_signal_tot = x_train_noshift_signal


# summerize events with 2D histogram
number_of_bins = 20
scale = 4
bins = np.linspace(-scale, scale, number_of_bins)

hist_x_train_signal = np.histogram2d(x_train_signal_tot[:, 1], x_train_signal_tot[:, 0], bins= [bins,bins])
hist_x_train_noshift_background = np.histogram2d(x_train_noshift_background[:, 1], x_train_noshift_background[:, 0], bins= [bins,bins])
hist_x_train_upshift_background = np.histogram2d(x_train_upshift_background[:, 1], x_train_upshift_background[:, 0], bins= [bins,bins])
hist_x_train_downshift_background = np.histogram2d(x_train_downshift_background[:, 1], x_train_downshift_background[:, 0], bins= [bins,bins])


def makeplot(histograms):
    limit = [-4, 4]
    plt.figure(figsize=(6, 6))
    cmap_sig = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0"] * 3)
    cmap_bkg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C1"] * 3)
    cmap = [cmap_sig, cmap_bkg, cmap_bkg, cmap_bkg]
    color=["C0", "C1", "C1",  "C1"]
    label=["Signal", "Background", "Background upshift", "Background downshift"]
    alpha = [0.8, 0.8, 0.4, 0.4]
    for i in range(0, len(histograms)):
        plt.contour(histograms[i][0], extent= [histograms[i][1][0], histograms[i][1][-1], histograms[i][2][0] , histograms[i][2][-1]], cmap=cmap[i], alpha=alpha[i])
        plt.plot([-999], [-999], color=color[i], label=label[i])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(limit[0], limit[1])
    plt.ylim(limit[0], limit[1])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0.)
    #plt.savefig("/home/risto/Masterarbeit/test.png", bbox_inches = "tight")
    plt.show()

makeplot([hist_x_train_signal, hist_x_train_noshift_background, hist_x_train_upshift_background, hist_x_train_downshift_background])

# save training data into pickle
pickle.dump([x_train_signal_tot, x_train_noshift_background, x_train_upshift_background, x_train_downshift_background], open("train.pickle", "wb"))

