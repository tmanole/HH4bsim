import ROOT
import numpy as np

bbbb_file = ROOT.TFile("../../events/ROOT16d/bbbb_run_02_toyTree.root","READ")
bbjj_file = ROOT.TFile("../../events/ROOT16d/bbjj_toyTree.root","READ")

bbbb_tree = bbbb_file.Get("Tree")
bbbb_tree.SetName("t4b")
bbjj_tree = bbjj_file.Get("Tree")
bbjj_tree.SetName("t2b")

n = 23570

lSR2b = []
lCR2b = []
lSB2b = []

lSR2bDer = []

vec2b = []
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())
vec2b.append(ROOT.TLorentzVector())


for i in range(0, n):
    bbjj_tree.GetEntry(i)
    
    bbjj_temp = []
    
    vec2b[0].SetPtEtaPhiE(bbjj_tree.jetPt[0], bbjj_tree.jetEta[0], bbjj_tree.jetPhi[0], bbjj_tree.jetEnergy[0])
    vec2b[1].SetPtEtaPhiE(bbjj_tree.jetPt[1], bbjj_tree.jetEta[1], bbjj_tree.jetPhi[1], bbjj_tree.jetEnergy[1])
    vec2b[2].SetPtEtaPhiE(bbjj_tree.jetPt[2], bbjj_tree.jetEta[2], bbjj_tree.jetPhi[2], bbjj_tree.jetEnergy[2])
    vec2b[3].SetPtEtaPhiE(bbjj_tree.jetPt[3], bbjj_tree.jetEta[3], bbjj_tree.jetPhi[3], bbjj_tree.jetEnergy[3])
    
    for j in range(0, 4):
        bbjj_temp.append(bbjj_tree.jetPt[j])
        #bbjj_temp.append(vec2b[j].Rapidity())   # Toggle this line to use rapidity instead of pseudorapidity
        bbjj_temp.append(bbjj_tree.jetEta[j])    # Toggle this line to use pseudorapidity.
        bbjj_temp.append(bbjj_tree.jetPhi[j])    
    
    if bbjj_tree.SR == 1:
        lSR2b.append(bbjj_temp)

    if bbjj_tree.CR == 1:
        lCR2b.append(bbjj_temp)
        
    if bbjj_tree.SB == 1:
        lSB2b.append(bbjj_temp)        
        
SB2b = np.array(lSB2b)        
CR2b = np.array(lCR2b)
SR2b = np.array(lSR2b)


m = 11690

lSR4b = []
lCR4b = []
lSB4b = []

vec4b = []
vec4b.append(ROOT.TLorentzVector())
vec4b.append(ROOT.TLorentzVector())
vec4b.append(ROOT.TLorentzVector())
vec4b.append(ROOT.TLorentzVector())

for i in range(m):
    bbbb_tree.GetEntry(i)
    
    bbbb_temp = []
    
    vec4b[0].SetPtEtaPhiE(bbbb_tree.jetPt[0], bbbb_tree.jetEta[0], bbbb_tree.jetPhi[0], bbbb_tree.jetEnergy[0])
    vec4b[1].SetPtEtaPhiE(bbbb_tree.jetPt[1], bbbb_tree.jetEta[1], bbbb_tree.jetPhi[1], bbbb_tree.jetEnergy[1])
    vec4b[2].SetPtEtaPhiE(bbbb_tree.jetPt[2], bbbb_tree.jetEta[2], bbbb_tree.jetPhi[2], bbbb_tree.jetEnergy[2])
    vec4b[3].SetPtEtaPhiE(bbbb_tree.jetPt[3], bbbb_tree.jetEta[3], bbbb_tree.jetPhi[3], bbbb_tree.jetEnergy[3])
    
    for j in range(4):
        bbbb_temp.append(bbbb_tree.jetPt[j])     
        #bbbb_temp.append(vec4b[j].Rapidity()) # Toggle this comment to use rapidity instead of pseudorapidity.
        bbbb_temp.append(bbbb_tree.jetEta[j])  # Toggle this line to use pseudorapidity.
        bbbb_temp.append(bbbb_tree.jetPhi[j])   
    
    if bbbb_tree.SR == 1:
        lSR4b.append(bbbb_temp)

    if bbbb_tree.CR == 1:
        lCR4b.append(bbbb_temp)
        
    if bbbb_tree.SB == 1:
        lSB4b.append(bbbb_temp) 

SB4b = np.array(lSB4b)
CR4b = np.array(lCR4b)
SR4b = np.array(lSR4b)


np.save("../../events/Numpy12d/SB2b.npy", SB2b)
np.save("../../events/Numpy12d/CR2b.npy", CR2b)
np.save("../../events/Numpy12d/SR2b.npy", SR2b)
np.save("../../events/Numpy12d/SB4b.npy", SB4b)
np.save("../../events/Numpy12d/CR4b.npy", CR4b)
np.save("../../events/Numpy12d/SR4b.npy", SR4b)


