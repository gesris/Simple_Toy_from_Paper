
import ROOT
import numpy as np

def get_data(filename):
    f = ROOT.TFile(filename, "READ")
    t = f.Get("limit")

    r = []
    deltaNLL = []
    for x in t:
        r.append(getattr(x, "r"))
        deltaNLL.append(getattr(x, "deltaNLL"))

    r = np.array(r)
    deltaNLL = 2.0 * np.array(deltaNLL)
    idx = np.argsort(r)
    r = r[idx]
    deltaNLL = deltaNLL[idx]
    return r, deltaNLL

r, deltaNLL = get_data("higgsCombinePlotNLL.MultiDimFit.mH120.root")
r_frozen, deltaNLL_frozen = get_data("higgsCombinePlotNLLFreezeSys.MultiDimFit.mH120.root")

plotNll = np.vstack((r, deltaNLL))
np.savetxt("plotNll.csv", plotNll)
plotNll_frozen = np.vstack((r_frozen, deltaNLL_frozen))
np.savetxt("plotNllFrozen.csv", plotNll_frozen)