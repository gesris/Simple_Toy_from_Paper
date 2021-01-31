import numpy as np
import matplotlib
matplotlib.rc("font", size=16, family="serif")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
np.random.seed(1234)
import pickle


# Make signal and background samples with up and down shifts in background process
def make_sig(num_events):
    mean = [0, 0]
    corr = -0.0
    cov = [[1.0, corr], [corr, 1.0]]
    return np.random.multivariate_normal(mean, cov, num_events)


def make_bkg(num_events, shift):
    mean = np.array([1, 1])
    mean = mean + shift
    corr = 0.0
    cov = [[1.0, corr], [corr, 1.0]]
    return np.random.multivariate_normal(mean, cov, num_events)


def make_dataset(num_events, shift):
    num_events = num_events
    x = np.vstack([make_sig(num_events), make_bkg(num_events, shift)])
    y = np.hstack([np.ones(num_events), np.zeros(num_events)])
    return x, y

def main(shift_scale, shift, plot_label):

    ####
    #### Create training dataset - important parameters for plots!
    ####

    num_train = 100000
    signal_exp = 50
    background_exp = 1000


    print("\nCreating data with shift: {}, shift scale: {}, plot label: {}".format(shift, shift_scale, plot_label))

    # Training dataset
    x_train, y_train = make_dataset(num_train, 0)
    x_train_up, _ = make_dataset(num_train, shift)
    x_train_down, _ = make_dataset(num_train, -shift)


    ##  weight for scaling
    signal_scale = signal_exp / float(num_train)
    background_scale = background_exp / float(num_train)

    w_train = np.ones(y_train.shape)
    w_train[y_train == 1] = signal_scale
    w_train[y_train == 0] = background_scale


    ## Testing dataset
    num_test = num_train
    x_test, y_test = make_dataset(num_test, 0)
    x_test_up, _ = make_dataset(num_test, shift)
    x_test_down, _ = make_dataset(num_test, -shift)

    w_test = np.ones(y_test.shape)
    w_test[y_test == 1] = signal_scale
    w_test[y_test == 0] = background_scale


    ##  Summerize events with 2D histogram
    number_of_bins = 20
    bins = (number_of_bins, number_of_bins)
    xlim = (-3, 5)
    ylim = (-3, 5)

    hist_x_train_signal = np.histogram2d(x_test[y_test==1][:, 1], x_test[y_test==1][:, 0], bins=bins, range=(xlim, ylim), density=True)
    hist_x_train_noshift_background = np.histogram2d(x_test[y_test==0][:, 1], x_test[y_test==0][:, 0], bins=bins, range=(xlim, ylim), density=True)
    hist_x_train_upshift_background = np.histogram2d(x_test_up[y_test==0][:, 1], x_test_up[y_test==0][:, 0], bins=bins, range=(xlim, ylim), density=True)
    hist_x_train_downshift_background = np.histogram2d(x_test_down[y_test==0][:, 1],x_test_down[y_test==0][:, 0], bins=bins, range=(xlim, ylim), density=True)


    def makeplot(histograms):
        limit = [-4, 4]
        plt.figure(figsize=(6, 6))
        cmap_sig = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C0"] * 3)
        cmap_bkg = matplotlib.colors.LinearSegmentedColormap.from_list("", ["C1"] * 3)
        cmap = [cmap_sig, cmap_bkg, cmap_bkg, cmap_bkg]
        color=["C0", "C1", "C1",  "C1"]
        label=["Signal", "Background", "Bkg. upshift", "Bkg. downshift"]
        ls = ["solid", "solid", "dashed", "dotted"]
        lw = 2
        alpha = [0.8, 0.8, 0.6, 0.6]
        for i in range(0, len(histograms)):
            plt.contour(histograms[i][0], extent= [histograms[i][1][0], histograms[i][1][-1], histograms[i][2][0] , histograms[i][2][-1]], cmap=cmap[i], linewidths=2, linestyles=ls[i], alpha=alpha[i])
            plt.plot([-999], [-999], color=color[i])
        plt.plot([-999], [-999], lw=lw, color="C0", label="Signal")
        plt.plot([-999], [-999], lw=lw, color="C1", label="Background")
        plt.plot([-999], [-999], lw=lw, ls="--", color="C1", alpha=0.8, label="Bkg. (up-shift)")
        plt.plot([-999], [-999], lw=lw, ls=":", color="C1", alpha=0.8, label="Bkg. (down-shift)")
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.savefig("./plots/sig_bkg_wave_{}".format(plot_label), bbox_inches = "tight")
        #plt.show()


    makeplot([hist_x_train_signal, hist_x_train_noshift_background, hist_x_train_upshift_background, hist_x_train_downshift_background])

    # save training data into pickle
    pickle.dump([x_train, x_train_up, x_train_down, y_train, w_train], open("train.pickle", "wb"))
    pickle.dump([x_train, x_test_up, x_test_down, y_test, w_test], open("test.pickle", "wb"))
    pickle.dump(plot_label, open("plot_label.pickle", "wb"))

if __name__ == "__main__":
    shift_scale = 1.0
    shift = shift_scale * np.array([0.0, 1.0])
    # labels sollten lauten: "CE_*", "SD_no_nuisance_*", "SD_with_nuisance_*"
    plot_label = "SD_with_nuisance2"
    main(shift_scale, shift, plot_label)
