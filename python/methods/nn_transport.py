import ROOT

from copy import copy
import numpy as np
from array import array

import sys
sys.path.insert(0, "../../transport_scripts/")
sys.path.insert(0, "../../")

from kernelized_transport import transport
from sklearn.neighbors import NearestNeighbors

from get_norm import get_norm

def nn_large_transport(bbbj,
                               bbbb,
                               out_path,
                               method_name,
                               coupling_path="../couplings/MG2/CR3b_SR3b/R0_4/coupling_block",
                               distance_path="../distances/MG2/emd_opt4/nn_CR3b_CR4b/tblock",
                               I_CR3b_hp="../couplings/MG2/ordering_sT/CR3b_SR3b/I_CR3b.npy",
                               I_SR3b_hp="../couplings/MG2/ordering_sT/CR3b_SR3b/I_SR3b.npy",
                               I_CR3b_vp="../distances/MG2/emd_opt4/nn_CR3b_CR4b/I_nn_CR3b.npy",#"../couplings/MG2/ordering_sT/CR3b_CR4b/I_CR3b.npy",
                               I_CR4b_vp="../distances/MG2/emd_opt4/nn_CR3b_CR4b/I_nn_CR4b.npy",#"../couplings/MG2/ordering_sT/CR3b_CR4b/I_CR4b.npy",
                               sT_vp="/media/tudor/Seagate Portable Drive/Seagate/LHC/distances/MG3/CR3b_CR4b/sT/sT",
                               nn_sliding_sT=True,
                               N_low_horiz=0, 
                               N_low_vert=0, 
                               N_high_horiz=16,
                               N_high_vert=11, 
                               bs=3, K=1, R0=2*np.pi, R=2):

    """ Transport from SR to CR in large dataset and apply FvT classifier. 
    
        "Source" and "target" are reversed from the usual notation. Source is typically SR3b, 
        and target is typically CR3b, for example.
    """

    w3b_CR_sum, w3b_SR_sum, w4b_CR_sum = get_norm(bbbj, bbbb)
    pi_factor = w4b_CR_sum / w3b_CR_sum

    # Create fit tree.  
    out_file = ROOT.TFile(out_path, "RECREATE")
    fit = bbbj.CloneTree(0)

    if "w_" + method_name not in bbbj.GetListOfBranches():
        fit.Branch("w_"+method_name, np.array([0], dtype=np.float32), "w_" + method_name + "/F")

    w = array('f',[0])
    fit.SetBranchAddress('w_'+method_name, w)

    I_CR3b_horiz = np.load(I_CR3b_hp).astype(int)
    I_SR3b_horiz = np.load(I_SR3b_hp).astype(int)

    I_CR3b_vert = np.load(I_CR3b_vp).astype(int)
    I_CR4b_vert = np.load(I_CR4b_vp).astype(int)

    nh = I_CR3b_horiz.shape[0]
    mh = I_SR3b_horiz.shape[0]

    nv = I_CR3b_vert.shape[0]
    mv = I_CR4b_vert.shape[0]

    nn_weights = np.zeros(bbbj.GetEntries())
    
    print(nn_weights.shape)
    print(N_high_vert)
    print(N_low_vert)
    print(np.max(I_CR3b_vert))

    nearest_neighbor=True

    print("============================================")
    print("Step 1: Starting NN lookup.")
    print("============================================")

    for j in range(N_low_vert, N_high_vert+1):
        print("block " + str(j))

        D = np.load(distance_path + str(j) + ".npy").transpose()

        if R != R0:
            P = np.load(sT_vp + str(j) + ".npy").transpose()

            D = (R0 * D/R) + ((R - R0) * P / R)

        if nearest_neighbor:
            idx = np.argsort(D, axis=1)[:, :K]

            overlap_4b = []
            overlap_3b = []

            if j > 0:
                overlap_4b = I_CR4b_vert[0:200 , j].tolist()
                overlap_3b = I_CR3b_vert[0:1500, j].tolist()

            for i in range(idx.shape[0]):         

                temp_weights = np.zeros(K)
                
                inds = I_CR3b_vert[idx[i, 0:K], j]
                s = 0

                for l in range(K):                    
                    if i in overlap_4b or inds[l] in overlap_3b:
                        temp_weights[l] = 2*D[i,idx[i,l]]

                    else:
                        temp_weights[l] = D[i,idx[i,l]]
                  
                s = np.sum(temp_weights)
                temp_weights = s/temp_weights              
                temp_weights /= np.sum(temp_weights)

                for l in range(K):                                      
                    nn_weights[inds[l]] += temp_weights[l] 
                    if nn_weights[inds[l]] > 25:
                        print("Exceeded 25 nearest neighbor hits: ", nn_weights[inds[l]], "inds: ", inds[l], ", ", idx[i,0])

#    print(nn_weights)
#    print(np.unique(nn_weights))
#    print(np.sum(nn_weights != 0)) 
#    print(np.sum(nn_weights != 0)/N) 

    print("============================================")
    print("Step 2: Starting horizontal transport.")
    print("============================================")

    weights = []
    wsum = 0

    for j in range(N_low_horiz, N_high_horiz+1):

        print("block " + str(j))

        ind_CR = I_CR3b_horiz[:, j].reshape([nh,1]).squeeze().tolist()

        coupling = np.load(coupling_path + str(j) + ".npy").transpose()

        normalize_marginal = np.diag(1.0/np.sum(coupling, axis=0))
        coupling =  np.matmul(coupling, normalize_marginal)

        weights.append(np.matmul(coupling, nn_weights[ind_CR]))
        wsum += np.sum(weights[-1]) 

        print(weights[-1])
        print(np.unique(weights[-1]))
        print(np.sum(weights[-1] != 0)/weights[-1].size) 
        print(np.max(weights[-1])) 
        print("Number of zeroes: ", np.sum(weights[-1] == 0))

    out_weights = np.repeat(0, bbbj.GetEntries()).tolist()

    for j in range(N_low_horiz, N_high_horiz+1):
        weights[j] = weights[j] * w3b_SR_sum *pi_factor/wsum
        ind_SR = I_SR3b_horiz[:, j].reshape([mh,1]).squeeze().tolist()

        for i in range(len(ind_SR)):
            out_weights[ind_SR[i]] = weights[j][i]

    i_SR = 0
    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w[0] = out_weights[i_SR]
            i_SR += 1

        else:
            w[0] = 0

        fit.Fill()

    out_file.Write()
    out_file.Close()
