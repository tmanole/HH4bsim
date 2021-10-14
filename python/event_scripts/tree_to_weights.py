import numpy as np
import argparse
import multiprocessing as mp
import imp
import ROOT


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='../../events/MG3/TTree/bbbj_SR.root', type=str, help='Path to first array, in .npy format.')
parser.add_argument('-o', '--output', default='../../events/MG3/weights/bbbj_SR.npy', type=str, help='Output path.')
args = parser.parse_args()

#print(args)

weights = []

f = ROOT.TFile(args.source, "READ")
tree = f.Get("Tree")

n = tree.GetEntries()
for j in range(n):
    tree.GetEntry(j)
    weights.append(tree.weight)

weights = np.array([weights]).T

np.save(args.output, weights)

