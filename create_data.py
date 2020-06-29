import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")
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
    x1 = np.vstack(make_sig(num_events))
    x2 = np.vstack(make_bkg(num_events, shift))
    y = np.hstack([np.ones(num_events), np.zeros(num_events)])
    return x1, x2, y

def main(shift_scale, shift, plot_label):
        

    ####
    #### Create training dataset - important parameters for plots!
    ####

    num_train = 100000
    signal_exp = 50
    background_exp = 1000
    signal_scale = signal_exp / float(num_train / 4.)
    background_scale = background_exp / float(num_train / 4.)


    print("\nCreating data with shift: {}, shift scale: {}, plot label: {}".format(shift, shift_scale, plot_label))

    # dataset with events containing each x-/y-coordinates
    x_train_noshift_signal, x_train_noshift_background, y_train = make_dataset(num_train, 0)
    x_train_upshift_signal, x_train_upshift_background, _ = make_dataset(num_train, shift)
    x_train_downshift_signal, x_train_downshift_background, _ = make_dataset(num_train, -shift)


    # weight for normalization
    w_train = np.ones(y_train.shape)
    w_train[y_train == 1] = signal_scale
    w_train[y_train == 0] = background_scale

    # summerize events with 2D histogram
    number_of_bins = 20
    scale = 4
    bins = np.linspace(-scale, scale, number_of_bins)

    hist_x_train_signal = np.histogram2d(x_train_noshift_signal[:, 1], x_train_noshift_signal[:, 0], bins= [bins,bins])
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
        plt.xlim(limit[0], limit[1])
        plt.ylim(limit[0], limit[1])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
        plt.savefig("./plots/sig_bkg_wave{}".format(plot_label), bbox_inches = "tight")
        #plt.show()


    makeplot([hist_x_train_signal, hist_x_train_noshift_background, hist_x_train_upshift_background, hist_x_train_downshift_background])

    # save training data into pickle
    pickle.dump([x_train_noshift_signal, x_train_upshift_signal, x_train_downshift_signal, x_train_noshift_background, x_train_upshift_background, x_train_downshift_background, y_train, w_train], open("train.pickle", "wb"))
    pickle.dump(plot_label, open("plot_label.pickle", "wb"))

if __name__ == "__main__":
    shift_scale = 1.0
    shift = shift_scale * np.array([0.0, 1.0])
    # labels sollten lauten: "CE_*", "SD_no_nuisance_*", "SD_with_nuisance_*"
    plot_label = "SD_with_nuisance_test2"
    main(shift_scale, shift, plot_label)
