import sys
sys.path.insert(0, "../../python/classifier/")
import make_df
import ROOT

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-pi', '--pathin',  default='../../events/MG2_small/TTree/bbbj_small_SR.root', type=str, help='Path to TTree.')
parser.add_argument('-po', '--pathout', default='../../events/MG2_small/dataframes/bbjj_small_SR.h5', type=str, help='Path to output directory.')
parser.add_argument('-f', '--fourTag', default=False, type=bool, help='True if even is four Tagged, false otherwise.')

args = parser.parse_args()

print(args.fourTag)

f = ROOT.TFile(args.pathin, "READ")
t = f.Get("Tree")
df = make_df.make_df(t, fromnp=False, fourTag=args.fourTag)
df.to_hdf(args.pathout, key="df", mode="w")

