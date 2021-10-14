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


bbbb_file = ROOT.TFile("../../events/MG3/TTree/bbbb.root","READ")
bbbj_file = ROOT.TFile("../../events/MG3/TTree/bbbj.root","READ")

bbbb_tree = bbbb_file.Get("Tree")
bbbb_tree.SetName("t4b")
bbbj_tree = bbbj_file.Get("Tree")
bbbj_tree.SetName("t3b")

CR3b_sT = []
CR4b_sT = []

for i in range(bbbj_tree.GetEntries()):
    bbbj_tree.GetEntry(i)
    
    if bbbj_tree.CR == 1:
        CR3b_sT.append(bbbj_tree.jetPt[0] + bbbj_tree.jetPt[1] + bbbj_tree.jetPt[2] + bbbj_tree.jetPt[3])

for i in range(bbbb_tree.GetEntries()):
    bbbb_tree.GetEntry(i)

    if bbbb_tree.CR == 1:
        CR4b_sT.append(bbbb_tree.jetPt[0] + bbbb_tree.jetPt[1] + bbbb_tree.jetPt[2] + bbbb_tree.jetPt[3])

N=34
n = len(CR3b_sT) #bbjj_tree.GetEntries()
m = len(CR4b_sT) #bbbb_tree.GetEntries()

print(n, m)

overlap3 = int(n/(N+1))
overlap4 = int(m/(N+1))

print(overlap4 == m/(N+1))

delta3 = 2 * overlap3
delta4 = 2 * overlap4

print("overlaps")
print(overlap3)
print(overlap4)

print("hypothetical")
print(2*int(n/7))
print(2*int(m/7))

#K_SR = 11500   # Desired block size along SR


# Overlaps

#n_CR = len(CR_sT)
#n_SR = len(SR_sT)
#
#K_SR = int(np.floor((n_SR/(N))))
#K_CR = int(np.floor((n_CR/(N)))) 

# Number of blocks
#N = int(np.floor(n_SR / K_SR))

#K_CR = int(np.floor(n_CR/N)) #+ overlap_CR
#K_SR += overlap_SR

print(N)

I_nn_CR3b = np.full((2*overlap3, N), -1)
I_nn_CR4b = np.full((2*overlap4, N), -1)

inds_CR3b = np.argsort(CR3b_sT)#[:N*K_CR]#.reshape([N, K_CR], order='F')
inds_CR4b = np.argsort(CR4b_sT)#[:N*K_SR]#.reshape([N, K_SR], order='F')

for i in range(N):
    I_nn_CR3b[:,i] = inds_CR3b[(i*overlap3):(i*overlap3 + (2*overlap3))] 
    I_nn_CR4b[:,i] = inds_CR4b[(i*overlap4):(i*overlap4 + (2*overlap4))] 

    print("start3: ", (i*delta3/2), "       end: ", (i*delta3/2 + delta3))
    print("start4: ", (i*delta4/2), "       end: ", (i*delta4/2 + delta4)) 

print(I_nn_CR3b.shape)
print(I_nn_CR4b.shape)

np.save("../../distances/MG3/nn_inds/I_nn_MG3_CR4b_halfstep.npy", I_nn_CR3b)
np.save("../../distances/MG3/nn_inds/I_nn_MG3_CR3b_halfstep.npy", I_nn_CR4b)
