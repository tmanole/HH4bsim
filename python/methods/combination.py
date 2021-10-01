import ROOT
from array import array
import sys
sys.path.insert(0, "../")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def resnet_transport_fit(bbbj_tree, df3b, df4b, classifier_path, coupling_cr_path, coupling_sr_path, out_path, bs=3):

    bid = str(bs) + "b"
    
    coupling_sr = np.load(coupling_sr_path).transpose()
    coupling_cr = np.load(coupling_cr_path).transpose()

    model = modelParameters(df3b[df3b['SB']], df4b[df4b['SB']], filename=classifier_path)

    pi = np.sum(df4b['SB'])/np.sum(df3b['SB'])

    h, _, _ = model.predict(df3b[df3b['SB']]) 

    weights_sr = 10000 * pi * np.matmul(coupling_sr, h/(1-h))
    weights_cr = 10000 * pi * np.matmul(coupling_cr, h/(1-h))
    weights_sb = 10000 * pi * h/(1-h)

    out_file = ROOT.TFile(out_path, "RECREATE") 
    fit = bbbj_tree.CloneTree(0)

    w = array('f',[0])
    fit.SetBranchAddress('weight', w)

    N = bbbj_tree.GetEntries()
    sr_count = 0
    cr_count = 0
    sb_count = 0
    for i in list(range(N)):
        bbbj_tree.GetEntry(i)
  
        if bbbj_tree.SR == 1:
            w[0] = weights_sr[sr_count]
            sr_count += 1

        elif bbbj_tree.CR == 1:
            w[0] = weights_cr[cr_count]
            cr_count += 1

        else:
            w[0] = weights_sb[sb_count]
            sb_count += 1 


        fit.Fill()

    out_file.Write()
    out_file.Close()


def resnet_large_transport(bbbj, bbbb, out_path, method_name, 
                           coupling_path, 
                           I_CR_path="../../../couplings/MG2/ordering_sT/I_CR3b.npy", 
                           I_SR_path="../../../couplings/MG2/ordering_sT/I_SR3b.npy", 
                           source="CR", N_low=0, N_high=16, bs=3, fvt=True, lrInit=0.01, train_batch_size=256, 
                           num_params=6):

    """ Transport from SR to CR in large dataset and apply FvT classifier. 
    
        "Source" and "target" are reversed from the usual notation. Source is typically SR3b, 
        and target is typically CR3b, for example.
    """

    bid = str(bs) + "b"

    I_CR = np.load(I_CR_path).astype(int)
    I_SR = np.load(I_SR_path).astype(int)

    n = I_CR.shape[0]
    m = I_SR.shape[0]

    weights = []

    out_file = ROOT.TFile(out_path, "RECREATE") 
    fit = bbbj.CloneTree(0)

    if "w_" + method_name not in bbbj.GetListOfBranches():
        fit.Branch("w_"+method_name, np.array([0], dtype=np.float32), "w_" + method_name + "/F")

    w = array('f',[0])
    fit.SetBranchAddress('w_'+method_name, w)

    weight_vec = []
    w_sum = 0

    full_h = []

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.CR == 1:
            full_h.append(bbbj.FvT)

    full_h=np.array(full_h)

###    if fvt:
###        sys.path.insert(0, "../../fvt_scripts/")
###        from model_train import modelParameters
###
###        for i in bbbj.GetEntries():
###            bbbj.GetEntry(i)
###
###            if bbbj.SR == 1:
###                full_h.append(bbbj.FvT)
###
####        model = modelParameters(df3b[df3b[source]], df4b[df4b[source]], fileName=classifier_path, lrInit=lrInit, train_batch_size=train_batch_size, num_params=num_params)
####
####        h1, _  = model.predict(df3b[df3b[source]])
####        full_h      = h1[:,1]
###
###
    w3b_CR_sum = 0.0
    w3b_SR_sum = 0.0
    w4b_CR_sum = 0.0

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.CR == 1:
            w3b_CR_sum += bbbj.weight

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w3b_SR_sum += bbbj.weight

    for i in range(bbbb.GetEntries()):
        bbbb.GetEntry(i)

        if bbbb.CR == 1:
            w4b_CR_sum += bbbb.weight

    pi = w4b_CR_sum/w3b_CR_sum

    for j in range(N_low, N_high+1):

        print("block " + str(j))

        ind = I_CR[:, j].reshape([n,1]).squeeze().tolist()
        h = full_h[ind]

        coupling = np.load(coupling_path + str(j) + ".npy")        
        coupling /= np.sum(coupling)
   
        print(coupling.shape)
        print(np.matmul((h/(1-h))/(np.sum(coupling, axis=1)), coupling).shape)
        print(np.sum(coupling, axis=1).shape)

        weights= pi * np.matmul(h/(1-h), coupling)/ (N_high-N_low+1)
        #weights=pi * np.matmul((h/(1-h))/(np.sum(coupling, axis=1)), coupling)/ (N_high-N_low+1)   
        #weights = np.sum(df3b['SR']) * pi * np.matmul(coupling, h/(1-h)) / (N_high-N_low+1)

        print(np.sum(h/(1-h)))
        print("weight sum: ", np.sum(weights))
        print("coupling sum: ", np.sum(coupling))
        print("fvt sum: ", np.sum(pi*h/(1-h)))
        w_sum += np.sum(weights)

        weight_vec.append(weights)

        print(weights)

    print("FINAL WEIGHT SUM: ", w_sum)
    print("FINAL WEIGHT SIZE: ", np.array(weight_vec).shape)

#    n1 = np.sum(df3b[df3b['SR']]['weight'])
#    m = np.sum(df4b[df4b['CR']]['weight'])
#    r = np.sum(df3b[df3b['CR']]['weight'])

    out_weights = np.repeat(0, bbbj.GetEntries()).tolist()
    
    for j in range(N_low, N_high+1):
        for i in range(I_SR.shape[0]):
            out_weights[I_SR[i,j]] = weight_vec[j][i] * w3b_SR_sum #n1 * m / r #(w_sum * r)
    
    o = 0
    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w[0] = out_weights[o]
            o += 1

        fit.Fill()

    out_file.Write()
    out_file.Close()

