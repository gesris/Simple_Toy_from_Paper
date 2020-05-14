import ROOT
import numpy as np

bins = np.loadtxt("bins.csv")
sig = np.loadtxt("sig.csv")
bkg = np.loadtxt("bkg.csv")
bkg_up = np.loadtxt("bkg_up.csv")
bkg_down = np.loadtxt("bkg_down.csv")
data_obs = sig + bkg

bins_vec = ROOT.std.vector("float")(len(bins))
for i in range(len(bins)):
    bins_vec[i] = bins[i]
tf = ROOT.TFile("shapes.root", "recreate")
for x, name in [
            (sig, "sig"),
            (bkg, "bkg"),
            (bkg_up, "bkg_sysUp"),
            (bkg_down, "bkg_sysDown"),
            (data_obs, "data_obs")
            ]:

    h = ROOT.TH1F(name, name, len(bins) - 1, bins_vec.data())
    for i in range(len(bins) - 1):
        h.SetBinContent(i + 1, x[i])
        h.SetBinError(i + 1, np.sqrt(x[i]))
    h.Write()
tf.Close()