import numpy as np
import argparse
import multiprocessing as mp
import imp

import scipy.spatial

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='../../events/MG2_small/PtEtaPhi/CR3b.npy', type=str, help='Path to first array, in .npy format.')
parser.add_argument('-o', '--output', default='../../events/MG2_small/sT/CR3b.npy', type=str, help='Output path.')
args = parser.parse_args()

print(args)

source = np.load(args.source)

output = args.output

n = source.shape[0]

sT = []

for j in range(n):
    ev = np.reshape(source[j,:], (4,3))
    sT.append(ev[0,0] + ev[1,0] + ev[2,0] + ev[3,0])

sT = np.array([sT]).T

np.save(output, sT)

