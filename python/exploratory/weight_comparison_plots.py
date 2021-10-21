import ROOT
import numpy as np
import sys
import pathlib
import os.path
from array import array

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

import seaborn as sns

#sys.path.insert(0, "transport_scripts")
#import transport_func4b as f4b
sys.path.insert(0, "..")
from definitions import *



text_size = 8

matplotlib.rc('xtick', labelsize=text_size) 
matplotlib.rc('ytick', labelsize=text_size) 
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

lw = 3
elw=0.3



def make_bivariate_weight_plots(tree):

    pathlib.Path("temp_files").mkdir(parents=True, exist_ok=True) 
    
    w_fvt  = []
    w_comb = []
    w_nn1  = []
    w_nn10 = []
    w_benchmark = []
    svb = []

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.SR == 1:
            w_fvt.append (tree.w_HH_FvT__cl_np799_l0_01_e10)
            w_comb.append(tree.w_HH_Comb_FvT__pl_emd_p1_R0_4__cl_np799_l0_01_e10)
            w_nn1.append (tree.w_HH_OT__pl_emd_p1_R0_4__K_1)
            w_nn10.append(tree.w_HH_OT__pl_emd_p1_R0_4__K_10)
            w_benchmark.append(tree.w_benchmark)
            svb.append(tree.SvB)

    np.save("temp_files/weights_fvt.npy", w_fvt)
    np.save("temp_files/weights_comb.npy", w_comb)
    np.save("temp_files/weights_nn1.npy", w_nn1)
    np.save("temp_files/weights_nn10.npy", w_nn10)
    np.save("temp_files/weights_benchmark.npy", w_benchmark)
    np.save("temp_files/svb.npy", svb)

redo_weights = False
if redo_weights:                              
    f = ROOT.TFile("../../events/MG3/TTree/bbbj.root")
    t = f.Get("Tree")
    t.Show(0)
    
    make_bivariate_weight_plots(t)
    
w_fvt       = np.load("temp_files/weights_fvt.npy" ) 
w_comb      = np.load("temp_files/weights_comb.npy") 
w_nn1       = np.load("temp_files/weights_nn1.npy" ) 
w_nn10      = np.load("temp_files/weights_nn10.npy") 
w_benchmark = np.load("temp_files/weights_benchmark.npy" )
svb         = np.load("temp_files/svb.npy" )


### TODO
s=svb
method = "HH-FvT"
w_standard = w_benchmark

w0 = w_fvt
w1 = w_comb
w2 = w_nn10
inds = np.argsort(svb)

#fig = matplotlib.pyplot.gcf()
#fig.set_size_inches(8, 4)
#
#fig.set_size_inches(8, 4)
#print("Unique: ", np.unique(w1).size)

##plt.clf()
##plt.plot(s[inds], w1[inds]-w0[inds], "o", ms=0.5)
##plt.axhline(y=0, color='black', linestyle='-')
##plt.xlabel("SvB")
##plt.ylabel("$\Delta$weight")
##plt.savefig("fvt_vs_" + method + "_diff_ordered.png")
##
###wA = w0[:151215]
###wB = w1[:151215]
###
###dd = {"x": s[inds], "y": w1[inds]-w0[inds]}
###dd2 = {"x": wA[np.logical_and(wB > 0, wB < 0.35)], "y":wB[np.logical_and(wB>0, wB < 0.35)]}
###sns.displot(dd2, x="x", y="y", binwidth=(0.01, 0.01), cbar=True)
###plt.show()
##
##plt.clf()
##plt.plot(s[inds], (w1[inds]-w0[inds])/w_standard[inds], "o", ms=0.5)
##plt.axhline(y=0, color='black', linestyle='-')
##plt.xlabel("SvB")
##plt.ylabel("($\Delta$weight)/(SR3b weight)")
##plt.savefig("fvt_vs_" + method + "_diff_standardized.png")

violin=False
if violin:
    sx = s[inds][:151200]
    w1y = (w1[inds] - w0[inds])[:151200]
    w2y = (w2[inds] - w0[inds])[:151200]
    n_bins =4 
    delta_bins = sx.size/n_bins
    bins = []
    b = 0
    
    names = ["[0, 1st Quartile)",
             "[1st Quartile, Median)",
             "[Median, 3rd Quartile)",
             "[3rd Quartile, 1]",
            ]
    
    for i in range(sx.size):
        bins.append(names[b])
        if i > (b+1)*delta_bins:
            b += 1
            print(b)
    
    bins = np.array(bins)
    print(w1y.shape, w2y.shape, bins.shape, np.repeat(1,w1y.size).shape)
    df = pd.DataFrame.from_dict({"Weight Difference": np.concatenate([w1y, w2y]), 
                                 "SvB": np.concatenate([bins,bins]), 
                                 "": np.concatenate([np.repeat("$w_{\mathrm{Comb}} - w_{\mathrm{FvT}}$",w1y.size), 
                                                     np.repeat("$w_{\mathrm{OT}}   - w_{\mathrm{FvT}}$",w2y.size)])})
    sns.violinplot(x="SvB", y="Weight Difference",hue="",data=df, inner=None)
    plt.axhline(lw=0.75, c="grey", linestyle="dashed")
    
    for i in range(4):
        plt.axvline(i + 0.5, color='grey', lw=0.75, linestyle="dashed")
    
    
    plt.savefig("compare_weights_violin.pdf")

#n_bins =4 
#delta_bins = sx.size/n_bins
#bins = []
#b = 0
#for i in range(sx.size):
#    bins.append(b)
#    if i > (b+1)*delta_bins:
#        b += 1
#        print(b)
#import pandas as pd
#df = pd.DataFrame.from_dict({"x": wy, "bin": np.array(bins)})
#
#sns.violinplot(x="bin", y="x",data= df)
#
#plt.savefig("try_vio.pdf")
#

else:
    
    wA = w0[:151215]
    wB = w1[:151215]
    wC = w2[:151215]
    cond1 = (wB > 0) & (wB < 0.3) #np.logical_and(wB>0, wB < 0.25)
    cond2 = (wA < 0.35) & (wC < 1.25)
    
    wA1 = wA[cond1]
    wB  = wB[cond1]
    
    wA2 = wA#[cond2]
    wC  = wC#[cond2]
    
    dd = {"x": s[inds], "y": w1[inds]-w0[inds]}
    dd2 = {"x": np.concatenate([wA1, wA2]), 
           "y": np.concatenate([wB, wC]), 
           "z": np.concatenate([np.repeat("HH-FvT v HH-Comb", wA1.size), np.repeat("HH-FvT v HH-OT (K=10)", wA2.size)])} 

    ddd = pd.DataFrame.from_dict(dd2) 
    print(dd2)
    
    
    g = sns.FacetGrid(ddd, col="z",sharey=False, sharex=False)
    g.map_dataframe(sns.histplot, "y", "x", bins=30, cbar=True)
    sns.displot(dd2, x="x", y="y", binwidth=(0.01, 0.01), cbar=True)
    
"""
def foo(data, color, **kwgs):
    sns.histplot(data, x="y", y="x", bins=30)

with sns.plotting_context(font_scale=5.5):
    g = sns.FacetGrid(ddd, col="z",sharey=False, sharex=False)

cbar_ax = g.fig.add_axes([.92, .3, .02, .4])

g = g.map_dataframe(foo, cbar_ax=cbar_ax)#, vmin=0, vmax=6000)

g.fig.subplots_adjust(right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
g.map_dataframe(sns.histplot, "y", "x", bins=30, cbar=True)
"""

line_position_y = [0.06, 0]
line_position_x = [0.35, 0.3]

i=0
for ax in g.axes.flat:
    x = np.linspace(line_position_x[i], line_position_y[i], 1000)
    y = x
    ax.plot(x, y, '-r', ms=2, c="purple")
    i+=1

g.axes[0,0].set_xlabel('HH-Comb Weights')# \n (a)')
g.axes[0,1].set_xlabel('HH-NN Weights (K=10)')# \n (b)')
g.axes[0,0].set_ylabel('HH-FvT Weights')
g.set_titles("") 
g.savefig("compare_weights_heatmap.pdf")


#ddds   =  [pd.DataFrame.from_dict({"x": wA1, "y": wB}),
#           pd.DataFrame.from_dict({"x": wA2, "y": wC})]
#
#fig, axn = plt.subplots(2, 1, sharex=True, sharey=True)
#cbar_ax = fig.add_axes([.91, .3, .03, .4])
#
#for i, ax in enumerate(axn.flat):
#    sns.histplot(ddds[i], x="y", y="x", ax=ax,
#                cbar=i == 0,
#                vmin=0, vmax=6000,
#                cbar_ax=None if i else cbar_ax)
#
#fig.tight_layout(rect=[0, 0, .9, 1])
#fig.savefig("sns_try2.pdf")


##plt.clf()
##plt.hist2d(w0[:151215], w1[:151215], bins=[110,110], cmap=plt.get_cmap("cividis"))#, cmin=30)
##plt.xlabel("HH-Fvt Weight")
##
##if method == "nn":
##    plt.ylabel("HH-OT Weight")
##
##if method == "benchmark":
##    plt.ylabel("SR3b Weight")
##
##if method == "rf":
##    plt.ylabel("HH-RF Weight")
##
##if method == "comb":
##    plt.ylabel("HH-Comb Weight")
##
##
##if method != "nn":
##   plt.xlim([0.06, 0.25])
##   plt.ylim([0.06, 0.25])
##
##else:
##   plt.xlim([0.02, 0.5])
##   plt.ylim([-0.01, 7])
##
##plt.colorbar()
##
##x = np.linspace(0.06,0.5,1000)
##y = x
##plt.plot(x, y, '-r', ms=3)
##figure(figsize=(8,8))
##
##fig = matplotlib.pyplot.gcf()
##fig.set_size_inches(8, 8)
##plt.show()
##plt.savefig("fvt_vs_" + method + "_biv_ordered.png")
##sys.exit()
##
##plt.clf()
##
##
##
