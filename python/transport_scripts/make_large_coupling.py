import numpy as np
import ot
import argparse

parser = argparse.ArgumentParser(description='')
#parser.add_argument('-s', '--sourcepath', default='../../events/MG3/PtEtaPhi/CR3b.npy', type=str, help='Path to source data.')
#parser.add_argument('-t', '--targetpath', default='../../events/MG3/PtEtaPhi/SR3b.npy', type=str, help='Path to target data.')
parser.add_argument('-sw', '--sweightpath', default='../../events/MG3/weights/bbbj_CR.npy', type=str, help='Path to source weights.')
parser.add_argument('-tw', '--tweightpath', default='../../events/MG3/weights/bbbj_SR.npy', type=str, help='Path to target weights.')
parser.add_argument('-d', '--distpath', default='../emd_scripts/MG3_blocks/CR3b_SR3b/tblock', type=str, help='Path to distance matrix.')
parser.add_argument('-sind', '--sourceind', default='../emd_scripts/MG3_blocks/CR3b_SR3b/I_MG3_CR3b.npy', type=str, help='Path to distance matrix.')
parser.add_argument('-tind', '--targetind', default='../emd_scripts/MG3_blocks/CR3b_SR3b/I_MG3_SR3b.npy', type=str, help='Path to distance matrix.')
parser.add_argument('-rl', '--rangelow', default=0, type=int, help='Path to distance matrix.')
parser.add_argument('-rh', '--rangehigh', default=16, type=int, help='Path to distance matrix.')
parser.add_argument('-o', '--output', default= '../../couplings/MG3/CR3b_SR3b/R0_4/coupling_block', type=str, help='Output path.')
parser.add_argument('-iter', '--maxiter', default= 1e7, type=float, help='Maximum number of iterations of the Hungarian algorithm.')
parser.add_argument('-R0', '--R0', default=2*np.pi, type=float, help='Default tuning parameter for the distance.')
parser.add_argument('-R', '--R', default=0.4, type=float, help='Tuning parameter for the distance.')
parser.add_argument('-sTp', '--sTpath', default="../emd_scripts/MG3_blocks/sT/sT", type=str, help='Path to sT difference matrices.')

args = parser.parse_args()

range_low  = args.rangelow
range_high = args.rangehigh
    
#source = np.load(args.sourcepath)
#target = np.load(args.targetpath)

source_weights = np.load(args.sweightpath)
target_weights = np.load(args.tweightpath)

source_weights /= np.sum(source_weights)
target_weights /= np.sum(target_weights)

source_weights = source_weights.reshape([-1])
target_weights = target_weights.reshape([-1])

I_source = np.load(args.sourceind).astype(int)
I_target = np.load(args.targetind).astype(int)

R0 = args.R0
R  = args.R

n = I_source.shape[0]
m = I_target.shape[0]

distance_path = args.distpath

sTpath = args.sTpath

outpath = args.output

for j in range(range_low, range_high+1):

    M = np.load(distance_path + str(j) + ".npy")

    if R != R0:
        S = np.load(sTpath + str(j) + ".npy")

        M = (R0 * M)/R + ((R - R0) * S/R)

        print("M changed")

#    x = source[I_source[:,j].reshape([n,1]).squeeze(),:]
#    y = target[I_target[:,j].reshape([m,1]).squeeze(),:]

    pi_s = source_weights[I_source[:,j].reshape([n,1]).squeeze()]
    pi_t = target_weights[I_target[:,j].reshape([m,1]).squeeze()]

    pi_s /= np.sum(pi_s)
    pi_t /= np.sum(pi_t)

    out = ot.emd(pi_s, pi_t, M = M, numItermax=1e7)

    np.save(outpath + str(j) + ".npy", out)




