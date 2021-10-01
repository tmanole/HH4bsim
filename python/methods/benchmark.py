import ROOT
from array import array
import numpy as np
import sys

def benchmark(bbbj, bbbb, out_path):

    out_file = ROOT.TFile(out_path, "RECREATE")
    fit = bbbj.CloneTree(0)

    if "w_benchmark" not in bbbj.GetListOfBranches():
        fit.Branch("w_benchmark", np.array([0], dtype=np.float32), "w_benchmark/F")

    w = array('f',[0])
    fit.SetBranchAddress('w_benchmark', w)

    w3b_sum = 0.0
    w4b_sum = 0.0
    for i in range(bbbb.GetEntries()):
        bbbb.GetEntry(i)

        if bbbb.CR == 1:
            w4b_sum += bbbb.weight

    for i in range(bbbj.GetEntries()):
        bbbj.GetEntry(i)

        if bbbj.CR == 1:
            w3b_sum += bbbj.weight

    pi = w4b_sum/w3b_sum 

    for i in list(range(bbbj.GetEntries())):
        bbbj.GetEntry(i)
        w[0] = pi * bbbj.weight
        fit.Fill()

    out_file.Write()
    out_file.Close()
