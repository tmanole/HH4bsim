from energyflow.emd import emd, emds
import numpy as np
import argparse
import emdFullCalc
import thrust
import time
import multiprocessing as mp

import imp
import ROOT
import numpy as np

#bbbb_file = ROOT.TFile("../../events/MG2/TTree/bbbb.root","READ")
bbbj_file = ROOT.TFile("../../events/MG3/TTree/bbbj.root","READ")

#bbbb_tree = bbbb_file.Get("Tree")
#bbbb_tree.SetName("t4b")
bbjj_tree = bbbj_file.Get("Tree")
bbjj_tree.SetName("t3b")

CR_sT = []
SR_sT = []

for i in range(bbjj_tree.GetEntries()):
    bbjj_tree.GetEntry(i)
    
    if bbjj_tree.SR == 1:
        SR_sT.append(bbjj_tree.jetPt[0] + bbjj_tree.jetPt[1] + bbjj_tree.jetPt[2] + bbjj_tree.jetPt[3])
    
    if bbjj_tree.CR == 1:
        CR_sT.append(bbjj_tree.jetPt[0] + bbjj_tree.jetPt[1] + bbjj_tree.jetPt[2] + bbjj_tree.jetPt[3])


#for i in range(bbbb_tree.GetEntries()):
#    bbbb_tree.GetEntry(i)
#
#    if bbbb_tree.CR == 1:
#        CR_sT.append(bbbb_tree.jetPt[0] + bbbb_tree.jetPt[1] + bbbb_tree.jetPt[2] + bbbb_tree.jetPt[3])

K_SR = 8895#9000   # Desired block size along SR

n_CR = len(CR_sT)
n_SR = len(SR_sT)

# Number of blocks
N = int(np.floor(n_SR / K_SR))

K_CR = int(np.floor(n_CR/N))

print(K_SR)
print(K_CR)
print(N)


inds_CR = np.argsort(CR_sT)
inds_SR = np.argsort(SR_sT)

I_CR = np.empty((K_CR, N))

k = 0
for i in range(K_CR):
    for j in range(N):
        I_CR[i, j] = inds_CR[k]
        k += 1

I_SR = np.empty((K_SR, N))

k = 0
for i in range(K_SR):
    for j in range(N):
        I_SR[i, j] = inds_SR[k]
        k += 1


np.save("I_MG3_CR3b.npy", I_CR)
np.save("I_MG3_SR3b.npy", I_SR)

print(I_CR.shape)
print(I_SR.shape)
