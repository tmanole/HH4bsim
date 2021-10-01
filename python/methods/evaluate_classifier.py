import ROOT
from array import array
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, "../../")

from get_norm import get_norm

def fit(bbbj, bbbb, method_name, out_path, source, target, fvt=True, lrInit=0.01, train_batch_size=512, num_params=6):
    
    w3b_CR_sum, w3b_SR_sum, w4b_CR_sum = get_norm(bbbj, bbbb)
    pi = w4b_CR_sum / w3b_CR_sum

    h = []
    h_true = []

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.SR == 1 or bbbj.CR == 1:
            h.append(bbbj.FvT)

    for j in range(bbbb.GetEntries()):
        bbbb.GetEntry(j)

        if bbbb.SR == 1 or bbbb.CR == 1:
            h_true.append(bbbb.FvT)

    h=np.array(h)
    h_true=np.array(h_true)

    weights      = h/(1-h)
    norm_weights = weights * w4b_CR_sum / w3b_CR_sum 

    weights_true = h_true/(1-h_true)


    # Create fit tree.  
    out_file = ROOT.TFile(out_path, "RECREATE")
    fit = bbbj.CloneTree(0)

    if "w_" + method_name not in bbbj.GetListOfBranches():
        fit.Branch("w_"+method_name, np.array([0], dtype=np.float32), "w_" + method_name + "/F")

    w = array('f',[0])
    fit.SetBranchAddress('w_'+method_name, w)

    N = bbbj.GetEntries()
    ctr = 0
    s_ctr = 0
    
    target=["CR","SR"]

    for i in list(range(N)):
        bbbj.GetEntry(i)
 
        b1 = bbbj.SB == 1 and "SB" in target
        b2 = bbbj.CR == 1 and "CR" in target
        b3 = bbbj.SR == 1 and "SR" in target

        if b1 or b2 or b3:
            w[0] = bbbj.weight * norm_weights[ctr]
            ctr += 1        
            fit.Fill()
 
    out_file.Write()
    out_file.Close()
    
#    N = bbbb.GetEntries()
#    ctr = 0
#    s_ctr = 0
#    for i in range(N):
#        bbbb.GetEntry(i)
# 
#        b1 = bbbb.SB == 1 and "SB" in target
#        b2 = bbbb.CR == 1 and "CR" in target
#        b3 = bbbb.SR == 1 and "SR" in target
# 
#        if b1 or b2 or b3:
#            truth.Fill()
#            ctr += 1    

    out_file.Write()
    out_file.Close()
