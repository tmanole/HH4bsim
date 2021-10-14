import ROOT
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin',  default='../../events/MG2/TTree/bbbb.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG2_medium/TTree/bbbb.root', type=str, help='Path to output directory.')
parser.add_argument('-N',  '--N', default=5000, type=int, help='Number of entries to select.')

args = parser.parse_args()

tree_file = ROOT.TFile(args.pathin, "READ")
tree = tree_file.Get("Tree")

new_file = ROOT.TFile(args.pathout, 'RECREATE')
new_tree = tree.CloneTree(args.N)

indices = np.random.randint(tree.GetEntries(), size=args.N)

print(tree.GetEntries())
print(indices[0:200])

for i in indices:
    tree.GetEntry(i)
#    new_tree.Fill()

print(new_tree.GetEntries())
    

new_file.Write()
new_file.Close()


