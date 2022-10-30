import ROOT

import numpy as np
import ot
import os

import matplotlib.pyplot as plt

def plot_coupling(sourcepath, targetpath, couplingpath, source_inds=None, target_inds=None, sreg='SB', treg='CR', outpath="plot.pdf", sT=True, marginalpath=None):

    coupling = np.load(couplingpath)
    
    sfile = ROOT.TFile(sourcepath, "READ")    
    source = sfile.Get("Tree")

    tfile = ROOT.TFile(targetpath, "READ")
    target = tfile.Get("Tree")

    n = source.GetEntries()

    s_sT = []

    j = 0
    for i in range(source.GetEntries()):
        source.GetEntry(i)
    
        if source_inds is not None and i not in source_inds:
            continue

        check = 0

        if sreg == 'SR' and source.SR == 1:
            s_sT.append(source.jetPt[0] + source.jetPt[1] + source.jetPt[2] + source.jetPt[3])
            check = 1

        if sreg == 'CR' and source.CR == 1:
            s_sT.append(source.jetPt[0] + source.jetPt[1] + source.jetPt[2] + source.jetPt[3])
            check = 1

        if sreg == 'SB' and source.SB == 1:
            s_sT.append(source.jetPt[0] + source.jetPt[1] + source.jetPt[2] + source.jetPt[3])
            check = 1

        if check > 0:
            j += 1

        if j >= n:
            break
    
    m = target.GetEntries()

    t_sT = []

    j = 0
    for i in range(target.GetEntries()):
        target.GetEntry(i)
    
        if target_inds is not None and i not in target_inds:
            continue

        check = 0

        if treg == 'SR' and target.SR == 1:
            t_sT.append(target.jetPt[0] + target.jetPt[1] + target.jetPt[2] + target.jetPt[3])
            check += 1

        if treg == 'CR' and target.CR == 1:
            t_sT.append(target.jetPt[0] + target.jetPt[1] + target.jetPt[2] + target.jetPt[3])
            check += 1

        if treg == 'SB' and target.SB == 1:
            t_sT.append(target.jetPt[0] + target.jetPt[1] + target.jetPt[2] + target.jetPt[3])
            check += 1

        if check > 0:
            j += 1

        if j >= n:
            break

    xinds = np.argsort(s_sT)
    yinds = np.argsort(t_sT)

    n = len(s_sT)
    m = len(t_sT)

    mind_x = xinds.reshape([n, 1])
    mind_y = yinds.reshape([1, m])

    s_sT_sort = np.array(s_sT)[xinds]
    t_sT_sort = np.array(t_sT)[yinds]

    coupling_sort = coupling[mind_x, mind_y]

    x = []
    y = []
    col = []

    cs = coupling_sort > 0
    ind = np.argwhere(cs)
    col = coupling_sort[cs]

    if sT:
        x = s_sT_sort[ind[:,0]]
        y = t_sT_sort[ind[:,1]]    

    else:
        x = ind[:,0]
        y = ind[:,1]    

    fig, ax = plt.subplots()


    scatter = ax.scatter(x, y, c=col, cmap = 'RdPu', alpha=0.5, s=0.1)

    if sT:
        plt.xlim((200, 600))
        plt.ylim((200, 600))

    plt.savefig(outpath)
    plt.clf()

    if marginalpath is not None:
        marginals = np.sum(coupling_sort, axis=0)

        plt.bar(range(marginals.size), marginals)
        plt.savefig(marginalpath)


def plot_coupling_wsT(s_sT, t_sT, couplingpath="../../couplings/MG2/CR3b_SR3b/R2/coupling_block0.npy", 
                      sreg='CR', 
                      treg='SR', 
                      outpath="CR_SR_gradient_sT.pdf", 
                      sT=True, 
                      marginalpath=None):

    coupling = np.load(couplingpath)

    xinds = np.argsort(s_sT)
    yinds = np.argsort(t_sT)

    print(s_sT)

    n = len(s_sT)
    m = len(t_sT)

    mind_x = xinds.reshape([n, 1])
    mind_y = yinds.reshape([1, m])
    print(mind_x)
    s_sT_sort = np.array(s_sT)[xinds]
    t_sT_sort = np.array(t_sT)[yinds]

    coupling_sort = coupling[mind_x, mind_y]

    x = []
    y = []
    col = []

    cs = coupling_sort > 0
    ind = np.argwhere(cs)
    col = coupling_sort[cs]


    print(ind)

    if sT:
        x = s_sT_sort[ind[:,0]]
        y = t_sT_sort[ind[:,1]]    

    else:
        x = ind[:,0]
        y = ind[:,1]    

    fig, ax = plt.subplots()


    scatter = ax.scatter(x, y, c=col, cmap = 'RdPu', alpha=0.5, s=0.1)

    if sT:
        plt.xlim((160,800))
        plt.ylim((160,800))

    plt.savefig(outpath)
    plt.clf()

    if marginalpath is not None:
        marginals = np.sum(coupling_sort, axis=0)

        plt.bar(range(marginals.size), marginals)
        plt.savefig(marginalpath)



def plot_simple_coupling(couplingpath, sreg='SB', treg='CR', outpath="plot.pdf"):

    coupling = np.load(couplingpath)

    n = coupling.shape[0]
    m = coupling.shape[1]

    c = coupling > 0#1e-5
    ind = np.argwhere(c)
    col = coupling[c]

    x = ind[:,0]
    y = ind[:,1]    

    import pandas as pd
    import seaborn as sns

    cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

    df = pd.DataFrame({'x': x, 'y': y, 'col':col})
    ax = sns.relplot(x="x", y="y", palette=cmap,
                         hue="col", s=0.5,
                         data=df, linewidth=0, legend='brief')

    leg = ax._legend
    for t in leg.texts:
        # truncate label text to 4 characters
        t.set_text(t.get_text()[:6])

    leg.set_title("Coupling Weight")

    leg.texts[0].set_text("")
    leg.texts[1].set_text("< 5e-05")
    leg.texts[2].set_text("    5e-05")
    leg.texts[3].set_text("    1e-04")
    leg.texts[4].set_text("> 1e-04")


    print(ax._legend)
    plt.savefig(outpath)
    plt.clf()

