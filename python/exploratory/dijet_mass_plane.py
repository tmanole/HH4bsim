import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 25.7,8.27

import seaborn as sns
import pandas as pd
import ot
import root_numpy as rn
import plotting
import ROOT


def dijet_mass_plane(tree, c=None, stats_title=None, meta=None, SR=1, CR=0, SB=0, llim = None, slim = None, fromnp=True, inds=None):
    """ Plot the diject mass plane for a given TTree.
    
        Args:
            tree(TTree): The tree.
            SR(bool): True if the dijet mass plane should be plotted in the signal region, false otherwise.
            CR(bool): True if the dijet mass plane should be plotted in the control region, false otherwise.
            SB(bool): True if the dijet mass plane should be plotted in the sideband region, false otherwise.
            llim(list): Lower and upper limits of leading jet masses to plot.
            slim(list): Lower and upper limits of sub-leading jet masses to plot.
            
        Returns:
            c: A ROOT TCanvas object.
            h: A ROOT 2d histogram (TH2f) object.
            ind: Indices of tree entries which were plotted.
    """
    if c is None:
        c=ROOT.TCanvas()

    if stats_title is None:
        stats_title=""

    if meta is None:
        meta = "mass plane; lead dijet mass [GeV]; subl dijet mass [GeV]"

    h=ROOT.TH2F(stats_title, meta, 70, 70, 190, 70, 70, 190)#"dijetMassPlane","mass plane; lead dijet mass [GeV]; subl dijet mass [GeV]", 30, 30, 180, 30, 30, 180)

    j0=ROOT.TLorentzVector()
    j1=ROOT.TLorentzVector()
    j2=ROOT.TLorentzVector()
    j3=ROOT.TLorentzVector()

    ind = []
    i = 0

    if inds is None:
        inds = range(tree.GetEntries())

    for e in inds:
        tree.GetEntry(e)

        if SR == 0 and tree.SR == 1: continue
        if CR == 0 and tree.CR == 1: continue
        if SB == 0 and tree.SB == 1: continue

        if fromnp:
            j0.SetPtEtaPhiE(tree.jetPt0, tree.jetEta0, tree.jetPhi0, tree.jetEnergy0)
            j1.SetPtEtaPhiE(tree.jetPt1, tree.jetEta1, tree.jetPhi1, tree.jetEnergy1)
            j2.SetPtEtaPhiE(tree.jetPt2, tree.jetEta2, tree.jetPhi2, tree.jetEnergy2)
            j3.SetPtEtaPhiE(tree.jetPt3, tree.jetEta3, tree.jetPhi3, tree.jetEnergy3)

        else:
            j0.SetPtEtaPhiE(tree.jetPt[0], tree.jetEta[0], tree.jetPhi[0], tree.jetEnergy[0])
            j1.SetPtEtaPhiE(tree.jetPt[1], tree.jetEta[1], tree.jetPhi[1], tree.jetEnergy[1])
            j2.SetPtEtaPhiE(tree.jetPt[2], tree.jetEta[2], tree.jetPhi[2], tree.jetEnergy[2])
            j3.SetPtEtaPhiE(tree.jetPt[3], tree.jetEta[3], tree.jetPhi[3], tree.jetEnergy[3])

        jets = [j0,j1,j2,j3]

        pairing = rootToy4b.doJetPairing(jets)

        if pairing is None:
            i += 1
            continue

        leadM = (jets[pairing[0][0]] + jets[pairing[0][1]]).M()
        sublM = (jets[pairing[1][0]] + jets[pairing[1][1]]).M()

        if llim is not None and slim is not None:
            if leadM < llim[0] or leadM > llim[1] or sublM < slim[0] or sublM > slim[1]:
                i+=1
                continue

        ind.append(i)
        h.Fill(leadM, sublM)
        i += 1

#    h.GetYAxis().SetRange(

    return c, h, ind

