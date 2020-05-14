import ROOT
import matplotlib
matplotlib.rc("font", size=16, family="serif")
#matplotlib.rc("text", usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import sys
import pickle

f = ROOT.TFile("postfit.root", "READ")

def plot(rfile, fit):
    plt.figure(figsize=(6, 6))
    d = rfile.Get("cat_" + fit)
    robs = d.Get("data_obs")
    rsig = d.Get("TotalSig")
    rbkg = d.Get("TotalBkg")
    sig = []
    bkg = []
    sig_err = []
    bkg_err = []
    obs = []
    bins = []
    for i in range(robs.GetNbinsX()):
        bins.append(robs.GetBinLowEdge(i + 1))
        obs.append(robs.GetBinContent(i + 1))
        sig.append(rsig.GetBinContent(i + 1))
        bkg.append(rbkg.GetBinContent(i + 1))
        sig_err.append(rsig.GetBinError(i + 1))
        bkg_err.append(rbkg.GetBinError(i + 1))
    bins.append(1.0)
    sig = np.array(sig)
    sig_err = np.array(sig_err)
    bkg = np.array(bkg)
    bkg_err = np.array(bkg_err)
    obs = np.array(obs)
    bins = np.array(bins)

    times = 10
    lw = 3
    c = bins[:-1] + 0.5 * (bins[1] - bins[0])
    C0 = "#1f77b4"
    C1 = "#ff7f0e"

    h_sig, _, _ = plt.hist(c, weights=sig * times, bins=bins, histtype="step", lw=lw, color=C0, ls="-")
    h_sig_up, _, _ = plt.hist(c, weights=(sig + sig_err) * times, bins=bins, histtype="step", lw=lw, color=C0, ls="--")
    h_sig_down, _, _ = plt.hist(c, weights=(sig - sig_err) * times, bins=bins, histtype="step", lw=lw, color=C0, ls="-.")

    h_bkg, _, _ = plt.hist(c, weights=bkg, bins=bins, histtype="step", lw=lw, color=C1, ls="-")
    h_bkg_up, _, _ = plt.hist(c, weights=bkg + bkg_err, bins=bins, histtype="step", lw=lw, color=C1, ls="--")
    h_bkg_down, _, _ = plt.hist(c, weights=bkg - bkg_err, bins=bins, histtype="step", lw=lw, color=C1, ls="-.")

    plt.plot([0], [0], ls="-", lw=lw, color=C0, label="Signal x{}".format(times))
    plt.plot([0], [0], ls="--", lw=lw, color=C0, label="Sig. (up-shift)")
    plt.plot([0], [0], ls="-.", lw=lw, color=C0, label="Sig. (down-shift)")
    plt.plot([0], [0], ls="-", lw=lw, color=C1, label="Background")
    plt.plot([0], [0], ls="--", lw=lw, color=C1, label="Bkg. (up-shift)")
    plt.plot([0], [0], ls="-.", lw=lw, color=C1, label="Bkg. (down-shift)")

    plt.xlabel("$f$")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0., prop={'size': 14})
    plt.ylabel("Count")
    plt.xlim(bins[0], bins[-1])
    plt.gca().ticklabel_format(axis="y", style="sci", scilimits=(0,3))
    plt.gca().get_yaxis().get_offset_text().set(va="top", ha="right")
    plt.savefig(fit + ".png", bbox_inches="tight")
    plt.savefig(fit + ".pdf", bbox_inches="tight")

plot(f, "prefit")
plot(f, "postfit")