import ROOT

from copy import copy
import numpy as np
from array import array

import sys
sys.path.insert(0, "../../transport_scripts/")
sys.path.insert(0, "../../")

#from kernelized_transport import transport
from sklearn.neighbors import NearestNeighbors

from get_norm import get_norm

def nn_large_transport(bbbj,
                       bbbb,
                       out_path,
                       method_name,
                       coupling_CR_path,
                       coupling_SR_path,
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
                       N_high_vert=33, 
                       bs=3, K=1, R0=2*np.pi, R=2.75):

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

    print("C3 c4 shape   ", I_CR3b_vert.shape, I_CR4b_vert.shape)

    nh = I_CR3b_horiz.shape[0]
    mh = I_SR3b_horiz.shape[0]

    nv = I_CR3b_vert.shape[0]
    mv = I_CR4b_vert.shape[0]

    overlap3b = 3551#int(nv/(N_high_vert+1))
    overlap4b = 484#int(mv/(N_high_vert+1))

    nn_weights = np.zeros(bbbj.GetEntries())
    
    print(nn_weights.shape)
    print(N_high_vert)
    print(N_low_vert)
    print(np.max(I_CR3b_vert))

    nearest_neighbor=True

    print("============================================")
    print("Step 1: Starting NN lookup.")
    print("============================================")
    full_inv_dists = np.zeros((16940, 124285), dtype=np.float32)  ## This is a very large matrix. 

    for j in range(N_low_vert, N_high_vert+1):
        print("block " + str(j))
        D = np.load(distance_path + str(j) + ".npy").transpose()

        if R != R0:
            P = np.load(sT_vp + str(j) + ".npy").transpose()

            print(P.shape, D.shape)

            D = (R0 * D/R) + ((R - R0) * P / R)

        print("Check: ", full_inv_dists[j*overlap4b:(j+1)*overlap4b, j*overlap3b:(j+1)*overlap3b] == 1.0/D)
        print("overlap:   ", ((j+2)*overlap4b) - (j*overlap4b))
        print(D.shape)
        print(full_inv_dists[(j*overlap4b):((j+2)*overlap4b), (j*overlap3b):((j+2)*overlap3b)].shape)
        full_inv_dists[(j*overlap4b):((j+2)*overlap4b), (j*overlap3b):((j+2)*overlap3b)] = 1.0/D

    D = 0 
    P = 0

    flattened_inds = I_CR3b_vert.flatten("F")
   
    flat_inds = np.empty(124285, dtype=np.int)
    for j in range(N_low_vert, N_high_vert+1):
        flat_inds[j*overlap3b: (j+1)*overlap3b] = flattened_inds[2*j*overlap3b: (2*j+1)*overlap3b]
        print((2*j+1)*overlap3b/flattened_inds.size)

    j = N_high_vert
    flat_inds[(j+1)*overlap3b:] = flattened_inds[(2*j+1)*overlap3b:]

    print(flat_inds)
    print(flat_inds.size)
    print(flattened_inds.size)
#    sys.exit()
    idx = np.argsort(-full_inv_dists, axis=1)[:, :K]

    print("exist sorting")


    for i in range(idx.shape[0]):         

        if i % 1000 == 0:
            print(i)

            #temp_weights = np.zeros(K)
        temp_weights = full_inv_dists[i,idx[i,:]]

        inds = flat_inds[idx[i, 0:K]]
              
        temp_weights /= np.sum(temp_weights)

        for l in range(K):                                      
            nn_weights[inds[l]] += temp_weights[l] 
#                if nn_weights[inds[l]] > 25:
#                    print("Exceeded 25 nearest neighbor hits: ", nn_weights[inds[l]], "inds: ", inds[l], ", ", idx[i,0])
        
        if np.max(inds) > 124285:
            print("Exceeded CR3b indexing")

    print(nn_weights)
    print(np.unique(nn_weights))
    print(np.sum(nn_weights != 0)) 
#    print(np.sum(nn_weights != 0)/N) 

    print("============================================")
    print("Step 2: Starting horizontal transport.")
    print("============================================")

    weights_CR = []
    weights_SR = []
    wsum_CR = 0
    wsum_SR = 0

    for j in range(N_low_horiz, N_high_horiz+1):

        print("block " + str(j))

        ind_CR = I_CR3b_horiz[:, j].reshape([nh,1]).squeeze().tolist()

        coupling_SR = np.load(coupling_SR_path + str(j) + ".npy").transpose()
        coupling_CR = np.load(coupling_CR_path + str(j) + ".npy").transpose()

        normalize_CR_marginal = np.diag(1.0/np.sum(coupling_CR, axis=0))
        coupling_CR = np.matmul(coupling_CR, normalize_CR_marginal)

        normalize_SR_marginal = np.diag(1.0/np.sum(coupling_SR, axis=0))
        coupling_SR = np.matmul(coupling_SR, normalize_SR_marginal)

        weights_CR.append(np.matmul(coupling_CR, nn_weights[ind_CR]))
        wsum_CR += np.sum(weights_CR[-1]) 

        weights_SR.append(np.matmul(coupling_SR, nn_weights[ind_CR]))
        wsum_SR += np.sum(weights_SR[-1]) 

        print(weights_SR[-1])
        print(np.unique(weights_SR[-1]))
        print(np.sum(weights_SR[-1] != 0)/weights_SR[-1].size) 
        print(np.max(weights_SR[-1])) 
        print("Number of zeros: ", np.sum(weights_SR[-1] == 0))

    out_weights_CR = np.repeat(0, bbbj.GetEntries()).tolist()
    out_weights_SR = np.repeat(0, bbbj.GetEntries()).tolist()

    for j in range(N_low_horiz, N_high_horiz+1):
        weights_CR[j] = weights_CR[j] * w3b_CR_sum * pi_factor/wsum_CR
        weights_SR[j] = weights_SR[j] * w3b_SR_sum * pi_factor/wsum_SR

        ind_CR = I_CR3b_horiz[:, j].reshape([nh,1]).squeeze().tolist()

        for i in range(len(ind_CR)):
            out_weights_CR[ind_CR[i]] = weights_CR[j][i]

        ind_SR = I_SR3b_horiz[:, j].reshape([mh,1]).squeeze().tolist()

        for i in range(len(ind_SR)):
            out_weights_SR[ind_SR[i]] = weights_SR[j][i]

    i_CR = 0
    i_SR = 0
    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w[0] = out_weights_SR[i_SR]
            i_SR += 1

        elif bbbj.CR == 1:
            w[0] = out_weights_CR[i_CR]
            i_CR += 1
        
        else:
            w[0] = 0

        fit.Fill()

    out_file.Write()
    out_file.Close()
