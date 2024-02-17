import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 25.7,8.27

import seaborn as sns
import pandas as pd
import ot
import root_numpy as rn
import plotting
import ROOT
from copy import copy

def doJetPairing(jets):
    #all possible pairings of the four jets:
    pairings = [[[0,1],[2,3]],
                [[0,2],[1,3]],
                [[0,3],[1,2]]]
    
    """dRs0 = [jets[pairing[0][0]].DeltaR(jets[pairing[0][1]]) for pairing in pairings]
    dRs1 = [jets[pairing[1][0]].DeltaR(jets[pairing[1][1]]) for pairing in pairings]
    dRs  = dRs0+dRs1
    idxMin = dRs.index(min(dRs))
    idxOth = (idxMin+3)%6"""

    #compute m4j to apply m4j dependent deltaRjj requirements (MDRs) on the pairings
    m4j = (jets[0]+jets[1]+jets[2]+jets[3]).M()

    #For each pairing, calculate if it passes or fails the MDRs.
    #  If it passes compute the mass difference between the dijets.
    #  Find the pairing with the smallest mass difference.
    minMassDifference = 1e6 #start with an arbitrary large value
    selectedPairing = None
    for pairing in pairings:
        #need to compute which dijet in this pairing has more Pt because different MDRs are applied to the leading and subleading dijets
        dijet0PtSum = jets[pairing[0][0]].Pt() + jets[pairing[0][1]].Pt()
        dijet1PtSum = jets[pairing[1][0]].Pt() + jets[pairing[1][1]].Pt()

        #sort dijets by their PtSum (leading dijet = "lead", subleading dijet = "subl"):
        if dijet0PtSum > dijet1PtSum:
            idxLead = 0
            idxSubl = 1
        else:
            idxLead = 1
            idxSubl = 0

        leadDeltaR = jets[pairing[idxLead][0]].DeltaR(jets[pairing[idxLead][1]])
        sublDeltaR = jets[pairing[idxSubl][0]].DeltaR(jets[pairing[idxSubl][1]])

        #passLeadMDR = (360/m4j - 0.5 < leadDeltaR) and (leadDeltaR < 653/m4j + 0.475) if m4j < 1250 else (leadDeltaR < 1)
        #passSublMDR = (235/m4j       < sublDeltaR) and (sublDeltaR < 875/m4j + 0.350) if m4j < 1250 else (sublDeltaR < 1)

        passLeadMDR = (360/m4j - 0.5 < leadDeltaR) and (leadDeltaR < 653/m4j + 0.977) if m4j < 1250 else (leadDeltaR < 1)
        passSublMDR = (235/m4j       < sublDeltaR) and (sublDeltaR < 875/m4j + 0.800) if m4j < 1250 else (sublDeltaR < 1)


        passMDRs = passLeadMDR and passSublMDR

        if not passMDRs: continue # this pairing failed the MDRs

        #this pairing passed the MDRs. Of the pairings which pass, keep the one with the smallest dijet mass difference
        lead = jets[pairing[idxLead][0]] + jets[pairing[idxLead][1]]
        subl = jets[pairing[idxSubl][0]] + jets[pairing[idxSubl][1]]

        massDifference = abs(lead.M()-subl.M())

        if massDifference < minMassDifference: # found a new smallest mass difference, update the minimum variable
            minMassDifference = massDifference
            selectedPairing   = [copy(pairing[idxLead]), copy(pairing[idxSubl])] # keep track of the best pairing, put lead at idx 0

    return selectedPairing
    

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

        pairing = doJetPairing(jets)

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

