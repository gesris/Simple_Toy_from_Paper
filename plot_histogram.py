import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", size=16, family="serif")


####
#### Plot histogram displaying significance
####

## Load data
s, b, b_up, b_down, bins = pickle.load(open("plot_histogram.pickle", "rb"))
plot_label = pickle.load(open("plot_label.pickle", "rb"))

# Save shifts to disk
for h, name in [
        (bins, "bins"),
        (s, "sig"),
        (b, "bkg"),
        (b_up, "bkg_up"),
        (b_down, "bkg_down")]:
    np.savetxt(name + ".csv", h)




n = len(s)
bins_for_plots_middle = []                  # Central Point of Bin
for i in range(0, len(bins) - 1):
    bins_for_plots_middle.append(bins[i] + (bins[i + 1] - bins[i]) / 2)
border = 0.5

print("\nPlotting histogram\n")
lw = 2
plt.figure(figsize=(6, 6))
plt.hist(bins_for_plots_middle, weights= [s[0], s[1]], bins= bins, histtype="step", label="Signal", lw=2, color="C0")
plt.hist(bins_for_plots_middle, weights= [b[0], b[1]], bins= bins, histtype="step", label="Backgorund", lw=2, color="C1")
plt.hist(bins_for_plots_middle, weights= [b_up[0], b_up[1]], bins= bins, ls="--", histtype="step", label="Backgorund", lw=2, color="C1")
plt.hist(bins_for_plots_middle, weights= [b_down[0], b_down[1]], bins= bins, ls=":", histtype="step", label="Backgorund", lw=2, color="C1")
plt.plot([0], [0], ls="-", lw=lw, color="C0", label="Signal")
plt.plot([0], [0], ls="-", lw=lw, color="C1", label="Background")
plt.plot([0], [0], ls="--", lw=lw, color="C1", label="Bkg. (up-shift)")
plt.plot([0], [0], ls=":", lw=lw, color="C1", label="Bkg. (down-shift)")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
plt.xlabel("$f$")
plt.ylabel("Count")
plt.xlim(bins[0], bins[-1])
#plt.axvline(x=border, ymin=0, ymax=n, color="r", linestyle= "dashed", lw=2)
plt.savefig("./plots/histogram_{}.png".format(plot_label), bbox_inches = "tight")
#plt.show()
