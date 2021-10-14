import ROOT
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin', default='../../events/MG2/TTree/bbbj.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG2/PtEtaPhiM/', type=str, help='Path to output directory.')
parser.add_argument('-bs', '--bquarks', default=3, type=int, help='Number of b-quarks (typically 2, 3 or 4).')

args = parser.parse_args()

tree_file = ROOT.TFile(args.pathin,"READ")

tree = tree_file.Get("Tree")
tree.SetName("t")

lSR2b = []
lCR2b = []
lSB2b = []

lSR2bDer = []

vec2b = []
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())


for i in range(tree.GetEntries()):
    tree.GetEntry(i)
    
    temp = []
    
    vec2b[0].SetPtEtaPhiE(tree.jetPt[0], tree.jetEta[0], tree.jetPhi[0], tree.jetEnergy[0])
    vec2b[1].SetPtEtaPhiE(tree.jetPt[1], tree.jetEta[1], tree.jetPhi[1], tree.jetEnergy[1])
    vec2b[2].SetPtEtaPhiE(tree.jetPt[2], tree.jetEta[2], tree.jetPhi[2], tree.jetEnergy[2])
    vec2b[3].SetPtEtaPhiE(tree.jetPt[3], tree.jetEta[3], tree.jetPhi[3], tree.jetEnergy[3])
    
    for j in range(0, 4):
        temp.append(tree.jetPt[j])
        temp.append(tree.jetEta[j])
        temp.append((tree.jetPhi[j] % (2*np.pi)))    
        temp.append(vec2b[j].M())    

    if tree.SR == 1:
        lSR2b.append(temp)

    if tree.CR == 1:
        lCR2b.append(temp)
        
    if tree.SB == 1:
        lSB2b.append(temp)        

    i += 1
        
SB2b = np.array(lSB2b)        
CR2b = np.array(lCR2b)
SR2b = np.array(lSR2b)

np.save(args.pathout + "/SB" + str(args.bquarks) + "b.npy", SB2b)
np.save(args.pathout + "/CR" + str(args.bquarks) + "b.npy", CR2b)
np.save(args.pathout + "/SR" + str(args.bquarks) + "b.npy", SR2b)


