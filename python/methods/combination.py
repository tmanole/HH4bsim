import ROOT
from array import array
import numpy as np
import sys
sys.path.insert(0, "../")

from get_norm import get_norm

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
                           coupling_CR_path, coupling_SR_path,
                           I_CR_path="../../../couplings/MG2/ordering_sT/I_CR3b.npy", 
                           I_SR_path="../../../couplings/MG2/ordering_sT/I_SR3b.npy", 
                           source="CR", N_low=0, N_high=16, bs=3, fvt=True, lrInit=0.01, train_batch_size=256, 
                           num_params=6):

    """ Transport from SR to CR in large dataset and apply FvT classifier. 
    
        "Source" and "target" are reversed from the usual notation. Source is typically SR3b, 
        and target is typically CR3b, for example.
    """
    
    # Setup
    w3b_CR_sum, w3b_SR_sum, w4b_CR_sum = get_norm(bbbj, bbbb)
    pi = w4b_CR_sum / w3b_CR_sum

    I_CR = np.load(I_CR_path).astype(int)
    I_SR = np.load(I_SR_path).astype(int)

    n = I_CR.shape[0]
    m = I_SR.shape[0]

    # Create fit tree
    out_file = ROOT.TFile(out_path, "RECREATE") 
    fit = bbbj.CloneTree(0)

    if "w_" + method_name not in bbbj.GetListOfBranches():
        fit.Branch("w_"+method_name, np.array([0], dtype=np.float32), "w_" + method_name + "/F")

    w = array('f',[0])
    fit.SetBranchAddress('w_'+method_name, w)

    # Get classifier weights
    full_h = []

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.CR == 1:
            full_h.append(bbbj.FvT)

    full_h=np.array(full_h)

    # Create HH-Comb-FvT weights
    weight_vec_CR = []
    weight_vec_SR = []

    for j in range(N_low, N_high+1):

        print("block " + str(j))

        ind = I_CR[:, j].reshape([n,1]).squeeze().tolist()
        h = full_h[ind]

        coupling_CR = np.load(coupling_CR_path + str(j) + ".npy")        
        #coupling_CR /= np.sum(coupling_CR)

        coupling_SR = np.load(coupling_SR_path + str(j) + ".npy")        
        #coupling_SR /= np.sum(coupling_SR)
   
        print(coupling_CR.shape)
        print(coupling_SR.shape)
#        print(np.matmul((h/(1-h))/(np.sum(coupling, axis=1)), coupling).shape)

        weights_CR = pi * np.matmul(h/(1-h), coupling_CR)/ (N_high-N_low+1)
        weights_SR = pi * np.matmul(h/(1-h), coupling_SR)/ (N_high-N_low+1)
        # Notice that `coupling` is implicitly normalized by its first marginal
        # and h/(1-h) is implicitly multiplied by this same quantity. 

#        print(np.sum(h/(1-h)))
#        print("weight sum: ", np.sum(weights))
#        print("coupling sum: ", np.sum(coupling))
#        print("fvt sum: ", np.sum(pi*h/(1-h)))

        weight_vec_CR.append(weights_CR)
        weight_vec_SR.append(weights_SR)

    out_weights_CR = np.repeat(0, bbbj.GetEntries()).tolist()
    out_weights_SR = np.repeat(0, bbbj.GetEntries()).tolist()
    
    for j in range(N_low, N_high+1):
        for i in range(I_SR.shape[0]):
            out_weights_SR[I_SR[i,j]] = weight_vec_SR[j][i] * w3b_SR_sum #n1 * m / r #(w_sum * r)

        for i in range(I_CR.shape[0]):
            out_weights_CR[I_CR[i,j]] = weight_vec_CR[j][i] * w3b_CR_sum #n1 * m / r #(w_sum * r)

    # Populate final tree
    
    j_CR = 0
    j_SR = 0
    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1:
            w[0] = out_weights_SR[j_SR]
            j_SR += 1

        elif bbbj.CR == 1:
            w[0] = out_weights_CR[j_CR]
            j_CR += 1

        else:
            w[0] = 0

        fit.Fill()

    out_file.Write()
    out_file.Close()

