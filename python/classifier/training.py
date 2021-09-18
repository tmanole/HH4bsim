import ROOT
from make_df import make_df
import time, os, sys
from glob import glob
from copy import copy
import numpy as np
np.random.seed(0)
import pandas as pd
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from sklearn.metrics import roc_curve, auc # pip/conda install scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
from networks import *
np.random.seed(0)#always pick the same training sample
torch.manual_seed(0)#make training results repeatable 

from model_train import modelParameters, loaderResults


import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b', '--background', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018A/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
parser.add_argument('-e', '--epochs', default=20, type=int, help='N of training epochs.')
parser.add_argument('-l', '--lrInit', default=1e-3, type=float, help='Initial learning rate.')
parser.add_argument('-p', '--pDropout', default=0.4, type=float, help='p(drop) for dropout.')
parser.add_argument(      '--layers', default=3, type=int, help='N of fully-connected layers.')
parser.add_argument('-n', '--nodes', default=32, type=int, help='N of fully-connected nodes.')
parser.add_argument('-c', '--cuda', default=1, type=int, help='Which gpuid to use.')
parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

n_queue = 20
eval_batch_size = 16384
print_step = 10
rate_StoS, rate_BtoB = None, None
barScale=200
barMin=0.5

class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

loadCycler = cycler()

# Run on gpu if available
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()', torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device", device, torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:", device)

#from networkTraining import *


classifier='FvT'
weight = 'weight'

#Simple ROC Curve plot function
def plotROC(train, val, name): #fpr = false positive rate, tpr = true positive rate
    lumiRatio = 140/59.6
    S = val.tpr*sum_wS*lumiRatio
    B = val.fpr*sum_wB*lumiRatio + 2.5
    sigma = S / np.sqrt(S+B)
    iMaxSigma = np.argmax(sigma)
    maxSigma = sigma[iMaxSigma]
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)

    plt.xlabel('Rate( '+signalName+' to '+signalName+' )')
    plt.ylabel('Rate( '+backgroundName+' to '+backgroundName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training")
    ax.plot(val  .tpr, 1-val  .fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation")
    ax.legend(loc='lower left')
    ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(val.roc_auc))

    if rate_StoS and rate_BtoB:
        ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
        ax.text(rate_StoS+0.03, rate_BtoB-0.025, "Cut Based WP \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)

        tprMaxSigma, fprMaxSigma, thrMaxSigma = val.tpr[iMaxSigma], val.fpr[iMaxSigma], val.thr[iMaxSigma]
        ax.scatter(tprMaxSigma, (1-fprMaxSigma), marker='o', c='#d34031')
        ax.text(tprMaxSigma+0.03, (1-fprMaxSigma)-0.025, "Optimal WP, "+classifier+" $>$ %0.2f \n (%0.2f, %0.2f), $%1.2f\sigma$ with 140fb$^{-1}$"%(thrMaxSigma, tprMaxSigma, (1-fprMaxSigma), maxSigma), bbox=bbox)

    f.savefig(name)
    plt.close(f)


def plotNet(train, val, name):
    orange='#ef8636'
    blue='#3b75af'
    yS_val  , yB_val   = val  .y_pred[val  .y_true==1], val  .y_pred[val  .y_true==0]
    wS_val  , wB_val   = val  .w     [val  .y_true==1], val  .w     [val  .y_true==0]
    sumW_val   = np.sum(wS_val  )+np.sum(wB_val  )
    yS_train, yB_train = train.y_pred[train.y_true==1], train.y_pred[train.y_true==0]
    wS_train, wB_train = train.w     [train.y_true==1], train.w     [train.y_true==0]
    sumW_train = np.sum(wS_train)+np.sum(wB_train)
    wS_val  , wB_val   = wS_val  /sumW_val  , wB_val  /sumW_val
    wS_train, wB_train = wS_train/sumW_train, wB_train/sumW_train
    fig = pltHelper.plot([yS_val, yB_val, yS_train, yB_train], 
                         [b/20.0 for b in range(21)],
                         "NN Output ("+classifier+")", "Arb. Units", 
                         weights=[wS_val, wB_val, wS_train, wB_train],
                         samples=["Validation "+signalName,"Validation "+backgroundName,"Training "+signalName,"Training "+backgroundName],
                         colors=[blue,orange,blue,orange],
                         alphas=[0.5,0.5,1,1],
                         linews=[2,2,1,1],
                         ratio=True,
                         ratioTitle=signalName+' / '+backgroundName,
                         ratioRange=[0,5])
    fig.savefig(name)
    plt.close(fig)


df2b = pd.read_hdf('../../dataframes/test_df2b.h5', key='df')
df4b = pd.read_hdf('../../dataframes/test_df4b.h5', key='df')

model = modelParameters(df2b, df4b)

#model initial state
model.trainSetup()

# Training loop
for _ in range(2): 
    model.runEpoch()

print()
print(">> DONE <<")
if model.foundNewBest: print("Best ROC AUC =", model.validation.roc_auc_best)

