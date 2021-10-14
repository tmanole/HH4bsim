from energyflow.emd import emd, emds
import numpy as np
import argparse
import emdFullCalc
import thrust
import time
import multiprocessing as mp

import imp
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
parser.add_argument('-t', '--target', default='../../events/MG3/PtEtaPhi/SR3b.npy', type=str, help='Path to second array, in .npy format.')
parser.add_argument('-is', '--indsource', default='MG3_blocks/CR3b_SR3b/I_MG3_CR3b.npy', type=str, help='Output file path.')
parser.add_argument('-it', '--indtarget', default='MG3_blocks/CR3b_SR3b/I_MG3_SR3b.npy', type=str, help='Output file path.')
parser.add_argument('-Nl', '--Nlow', default=0, type=int, help='Output file path.')
parser.add_argument('-Nh', '--Nhigh', default=2, type=int, help='Output file path.')
parser.add_argument('-o', '--output', default='MG3_blocks/CR3b_SR3b/', type=str, help='Output file path.')
parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
args = parser.parse_args()

print(args)

source = np.load(args.source)
target = np.load(args.target)

output = args.output

I_CR = np.load(args.indsource)
I_SR = np.load(args.indtarget)

N = I_CR.shape[1]
print(N)
N_low = args.Nlow
N_high = args.Nhigh

if N != I_SR.shape[1]:
    print("ATTENTION, CR AND SR DO NOT HAVE SAME NUMBER OF BLOCKS.")

R = 2 * np.pi

print("starting")

start_time = time.time()
n_proc = args.nproc

n = I_SR.shape[0]
m = I_CR.shape[0]
print(n)
def process_chunk(x, y, bound):   
    low = bound[0]
    up = bound[1]

    csize = up-low
    M = np.zeros((csize, n))

    for j in range(n):
        ev1     = np.reshape(y[j,:], (4,3))
        sumPz1  = np.sum(ev1[:,0] * np.sinh(ev1[:,1]))        
        tAxis1  = thrust.getThrustAxis(ev1)

        if j % 10 == 0:
            print(N_low, N_high, j)
            print("--- %s seconds ---" % (time.time() - start_time))

        for i in range(low, up):
            ev0    = np.reshape(x[i,:], (4,3))
            sumPz0 = np.sum(ev0[:,0] * np.sinh(ev0[:,1]))
    
            M[i-low, j] = emdFullCalc.emdFullAprox(ev0=ev0, ev1=ev1, R=R, sumPz0=sumPz0, sumPz1=sumPz1, tAxis1=tAxis1)

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

    np.save(output + "tblock" + str(k) + ".npy", block)

print(args)
print("--- %s seconds ---" % (time.time() - start_time))




