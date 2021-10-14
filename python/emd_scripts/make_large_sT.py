import numpy as np
import argparse
import multiprocessing as mp
import imp

import scipy.spatial

#try:
#    imp.find_module("ROOT")
#    import ROOT
#
#except ImportError:
#    import sys
#    sys.path.insert(0, "../../../../ROOT/build/lib")
#    import ROOT_copy as ROOT

#f = ROOT.TFile("blabla.root", "RECREATE")
#print(f)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', default='../../events/MG3/PtEtaPhi/CR3b.npy', type=str, help='Path to first array, in .npy format.')
parser.add_argument('-t', '--target', default='../../events/MG3/PtEtaPhi/CR3b.npy', type=str, help='Path to second array, in .npy format.')
parser.add_argument('-is', '--indsource', default="MG3_blocks/CR3b_SR3b/I_MG3_CR3b.npy", type=str, help='Output file path.')
parser.add_argument('-it', '--indtarget', default="MG3_blocks/CR3b_SR3b/I_MG3_CR3b.npy", type=str, help='Output file path.')
parser.add_argument('-Nl', '--Nlow', default=0, type=int, help='Output file path.')
parser.add_argument('-Nh', '--Nhigh', default=18, type=int, help='Output file path.')
parser.add_argument('-o', '--output', default='MG3_blocks/CR3b_CR3b/sT/', type=str, help='Output file path.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-sd','--sd', default=False, type=bool, help='Divide by standard deviation?')
args = parser.parse_args()


print(args)

source = np.load(args.source)
target = np.load(args.target)

output = args.output

I_CR = np.load(args.indsource)
I_SR = np.load(args.indtarget)

N_low = args.Nlow
N_high = args.Nhigh

N = I_CR.shape[1]

if N != I_SR.shape[1]:
    print("ATTENTION, CR AND SR DO NOT HAVE SAME NUMBER OF BLOCKS.")

print("starting")

n_proc = args.nproc
sd = args.sd

n = I_SR.shape[0]
m = I_CR.shape[0]

def process_chunk(x, y, bound):   
    low = bound[0]
    up = bound[1]

    csize = up-low
    M = np.zeros((csize, n))

    sT0 = []
    sT1 = []

    for j in range(n):
        ev = np.reshape(y[j,:], (4,3))
        sT1.append(ev[0,0] + ev[1,0] + ev[2,0] + ev[3,0])

    for i in range(low, up):
        ev = np.reshape(x[i,:], (4,3))
        sT0.append(ev[0,0] + ev[1,0] + ev[2,0] + ev[3,0])

    sT0 = np.array([sT0]).T
    sT1 = np.array([sT1]).T

    print(sT0.shape)

    # Note: "Euclidean" here is really just the absolute value. 

    if sd:    # Absolute value between sT values.
        M = scipy.spatial.distance.cdist(sT0, sT1, metric="euclidean")

    else:     # Absolute value between sT values, scaled by standard deviation.
        M = scipy.spatial.distance.cdist(sT0, sT1, metric="seuclidean")

    return M


Del = m//n_proc 
for k in range(N_low, N_high+1):
    print("===================================================")
    print("Starting Block " + str(k))
    print("===================================================")
    
    proc_chunks = []

    x = source[I_CR.astype(int)[:, k].T, :]
    y = target[I_SR.astype(int)[:, k].T, :]

    def process_block_chunk(bound):
        return process_chunk(x, y, bound)

    for i in range(n_proc):
        if i == n_proc-1:
            proc_chunks.append(( (n_proc-1) * Del, m) )

        else:
            proc_chunks.append(( (i*Del, (i+1)*Del ) ))

    with mp.Pool(processes=n_proc) as pool:
        proc_results = [pool.apply_async(process_block_chunk, args=(chunk,))
                        for chunk in proc_chunks]
        result_chunks = [r.get() for r in proc_results]

    block = np.vstack(result_chunks)

    np.save(output + "sT" + str(k) + ".npy", block)





