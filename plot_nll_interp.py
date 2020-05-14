import matplotlib
matplotlib.rc("font", size=16, family="serif")
matplotlib.rc("text", usetex=True)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
import sys
import pickle

def get_data(filename):
    x = np.loadtxt(filename)
    r = x[0, :]
    deltaNLL = x[1, :]
    return r, deltaNLL

r, deltaNLL = get_data("plotNll.csv")
r_frozen, deltaNLL_frozen = get_data("plotNllFrozen.csv")

def f(x):
    return np.interp(x, r, deltaNLL)

def f_frozen(x):
    return np.interp(x, r_frozen, deltaNLL_frozen)

bounds = (0, 2)
opt = minimize_scalar(f, bounds=bounds, method="bounded")
opt_frozen = minimize_scalar(f_frozen, bounds=bounds, method="bounded")

def g(x):
    return np.abs(f(x) - 1)

def g_frozen(x):
    return np.abs(f_frozen(x) - 1)

low = minimize_scalar(g, bounds=(bounds[0], opt.x), method="bounded")
up = minimize_scalar(g, bounds=(opt.x, bounds[1]), method="bounded")

low_frozen = minimize_scalar(g_frozen, bounds=(bounds[0], opt_frozen.x), method="bounded")
up_frozen = minimize_scalar(g_frozen, bounds=(opt_frozen.x, bounds[1]), method="bounded")

plt.figure(figsize=(6,6))

# Box sys + stat
#plt.plot([r.min(), low.x], [1, 1], lw=1, color="k")
#plt.plot([up.x, r.max()], [1, 1], lw=1, color="k")
plt.plot([low.x] * 2, [0, 1], lw=1, color="r")
plt.plot([up.x] * 2, [0, 1], lw=1, color="r")

# Box stat only
plt.plot([r_frozen.min(), low_frozen.x], [1, 1], lw=1, color="k")
plt.plot([up_frozen.x, r_frozen.max()], [1, 1], lw=1, color="k")
plt.plot([low_frozen.x] * 2, [0, 1], lw=1, color="b")
plt.plot([up_frozen.x] * 2, [0, 1], lw=1, color="b")

# Line sys + stat
r_ = np.linspace(r.min(), r.max(), 1000)
plt.plot(r_, f(r_), "-", lw=2, color="r", label="$\mu_\\mathrm{{sys.+stat.}} = {:.2f}\; ({:.2f}\;+{:.2f})$".format(opt.x, low.x - 1, up.x - 1))

# Line stat only
r_frozen_ = np.linspace(r_frozen.min(), r_frozen.max(), 1000)
plt.plot(r_frozen_, f_frozen(r_frozen_), "-", lw=2, color="b", label="$\mu_\\mathrm{{stat.\;\;\;\;\;\;\;\;\,}} = {:.2f}\; ({:.2f}\;+{:.2f})$".format(opt_frozen.x, low_frozen.x - 1, up_frozen.x - 1))

# Labels
plt.ylabel("$2\\cdot \\Delta\\mathrm{NLL}$")
plt.xlabel("$\mu$")

# Lims
plt.xlim(r.min(), r.max())
#plt.ylim(deltaNLL.min(), deltaNLL.max())
plt.ylim(0, 3.0)

# Legend
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=0., prop={'size': 16})

# Save
plt.savefig("plot_nll.png", bbox_inches="tight")
plt.savefig("plot_nll.pdf", bbox_inches="tight")