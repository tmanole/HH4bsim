import time, os, sys
import multiprocessing
from glob import glob
from copy import copy
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from nadam import NAdam
from sklearn.metrics import roc_curve, roc_auc_score, auc # pip/conda install scikit-learn
from roc_auc_with_negative_weights import roc_auc_with_negative_weights
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlibHelpers as pltHelper
#plt.rc('text', usetex=False) # uncomment this line if you can't get latex working
from scipy.stats import ks_2samp, chisquare

from networks import *
np.random.seed(0)#always pick the same training sample
torch.manual_seed(1)#make training results repeatable 

matplotlib.rcParams['figure.figsize'] = [8, 8]
plt.figure(figsize=(8,8))

##import argparse
##parser = argparse.ArgumentParser(description='Process some integers.')
##parser.add_argument('-d', '--data', default='/uscms/home/bryantp/nobackup/ZZ4b/data2018/picoAOD.h5',    type=str, help='Input dataset file in hdf5 format')
###parser.add_argument('-t', '--ttbar',      default='',    type=str, help='Input MC ttbar file in hdf5 format')
##parser.add_argument('-s', '--signal',     default='', type=str, help='Input dataset file in hdf5 format')
##parser.add_argument('-c', '--classifier', default='FvT', type=str, help='Which classifier to train: FvT, ZHvB, ZZvB, M1vM2.')
##parser.add_argument('-e', '--epochs', default=40, type=int, help='N of training epochs.')
##parser.add_argument('-o', '--outputName', default='', type=str, help='Prefix to output files.')
###parser.add_argument('-l', '--lrInit', default=4e-3, type=float, help='Initial learning rate.')
##parser.add_argument('-p', '--pDropout', default=0.4, type=float, help='p(drop) for dropout.')
##parser.add_argument(      '--layers', default=3, type=int, help='N of fully-connected layers.')
##parser.add_argument('-n', '--nodes', default=32, type=int, help='N of fully-connected nodes.')
##parser.add_argument('--cuda', default=1, type=int, help='Which gpuid to use.')
##parser.add_argument('-m', '--model', default='', type=str, help='Load this model')
##parser.add_argument(      '--onnx', dest="onnx",  default=False, action="store_true", help='Export model to onnx')
##parser.add_argument('-u', '--update', dest="update", action="store_true", default=False, help="Update the hdf5 file with the DNN output values for each event")
##parser.add_argument(      '--storeEvent',     dest="storeEvent",     default="0", help="store the network response in a numpy file for the specified event")
##parser.add_argument(      '--storeEventFile', dest="storeEventFile", default=None, help="store the network response in this file for the specified event")
###parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
##args = parser.parse_args()

import auc_ci

os.environ["CUDA_VISIBLE_DEVICES"]=str(1)##str(args.cuda)

##classifier = 'FvT'#args.classifier
# epochs = 40
pDropout = 0.4
layers = 3
nodes = 32
update_global = False
onnx = False
storeEvent = 0
storeEventFile = None
outputName = ""

n_queue = 0
eval_batch_size = 2**14#15
##train_batch_size = 2**10#11
##lrInit = 0.001#8e-1#4e-3
max_patience = 1
print_step = 2
rate_StoS, rate_BtoB = None, None
#barScale=200
#barMin=0.5
nClasses=2

class classInfo:
    def __init__(self, abbreviation='', name='', index=None, color=''):
        self.abbreviation = abbreviation
        self.name = name
        self.index = index
        self.color = color



d4 = classInfo(abbreviation='d4', name=  'FourTag Data',       index=0, color='red')
d3 = classInfo(abbreviation='d3', name= 'ThreeTag Data',       index=1, color='orange')
##t4 = classInfo(abbreviation='t4', name= r'FourTag $t\bar{t}$', index=2, color='green')
##t3 = classInfo(abbreviation='t3', name=r'ThreeTag $t\bar{t}$', index=3, color='cyan')

sg = classInfo(abbreviation='sg', name=  'Signal Data',       index=1, color='red')
bg = classInfo(abbreviation='bg', name= 'Background Data',    index=0, color='orange')

def getFrame(fileName):
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    print("Reading",fileName)
    thisFrame = pd.read_hdf(fileName, key='df')
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    return thisFrame


def getFrameSvB(fileName):
    print("Reading",fileName)    
    yearIndex = fileName.find('201')
    year = float(fileName[yearIndex:yearIndex+4])
    thisFrame = pd.read_hdf(fileName, key='df')
    fourTag = False if "data201" in fileName else True
    thisFrame = thisFrame.loc[ (thisFrame[trigger]==True) & (thisFrame['fourTag']==fourTag) & ((thisFrame['SB']==True)|(thisFrame['CR']==True)|(thisFrame['SR']==True)) & (thisFrame.FvT>0) ]#& (thisFrame.passXWt) ]
    thisFrame['year'] = pd.Series(year*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    thisFrame['target']  = pd.Series(index*np.ones(thisFrame.shape[0], dtype=np.float32), index=thisFrame.index)
    return thisFrame

#fileReaders = multiprocessing.Pool(10)


class cycler:
    def __init__(self,options=['-','\\','|','/']):
        self.cycle=0
        self.options=options
        self.m=len(self.options)
    def next(self):
        self.cycle = (self.cycle + 1)%self.m
        return self.options[self.cycle]

class nameTitle:
    def __init__(self,name,title):
        self.name = name
        self.title= title

loadCycler = cycler()


##wC = torch.FloatTensor([1, 1, 1, 1]).to("cuda")
###wC = torch.FloatTensor([1, 1, 1, 1]).to("cpu")
wC = torch.FloatTensor([1, 1]).to("cpu")





# Run on gpu if available
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
print('torch.cuda.is_available()',torch.cuda.is_available())
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Found CUDA device",device,torch.cuda.device_count(),torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU:",device)

#from networkTraining import *

class roc_data:
    def __init__(self, y_true, y_pred, weights, trueName, falseName):
        self.fpr, self.tpr, self.thr = roc_curve(y_true, y_pred, sample_weight=weights)
        self.auc = roc_auc_with_negative_weights(y_true, y_pred, weights=weights)
        self.trueName  = trueName
        self.falseName = falseName
        wS = 0
        self.maxSigma=0
##        if "Background" in self.falseName:
##            lumiRatio = 1#140/59.6
##            if "ZZ" in self.trueName: 
##                wS = wzz
##                self.pName = "P($ZZ$)"
##            if "ZH" in self.trueName: 
##                wS = wzh
##                self.pName = "P($ZH$)"
##            if "Signal" in self.trueName: 
##                wS = wzz+wzh
##                self.pName = "P(Signal)"
##            self.S = self.tpr*wS*lumiRatio
##            self.S[self.tpr<0.10] = 0 # require at least 10% signal acceptance 
##            self.B = self.fpr*sum_wB*lumiRatio
##            self.B[self.fpr<0.001] = 1.0e9 # require at least 0.1% background acceptance 
##            sigma = self.S / np.sqrt(self.S+self.B+2.5)
##            self.iMaxSigma = np.argmax(sigma)
##            self.maxSigma = sigma[self.iMaxSigma]
##            self.S = self.S[self.iMaxSigma]
##            self.B = self.B[self.iMaxSigma]
##            self.tprMaxSigma, self.fprMaxSigma, self.thrMaxSigma = self.tpr[self.iMaxSigma], self.fpr[self.iMaxSigma], self.thr[self.iMaxSigma]


class loaderResults_SvB:
    def __init__(self, name):
        self.name = name
        self.trainLoaders= []
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.n      = None
        self.w      = None
        self.roc= None #[0 for cl in classes]
        self.loss = 1e6
        self.loss_min = 1e6
        self.loss_prev = None
        self.loss_best = 1e6
        self.roc_auc_best = None
        self.sum_w_S = None
        self.norm_d4_over_B = 0

    def splitAndScale(self):
        self.psg = self.y_pred[:,sg.index]
        self.pbg = self.y_pred[:,bg.index]
##            self.ptt = self.y_pred[:,tt.index]
##            self.pmj = self.y_pred[:,mj.index]

##            self.pbg = self.pmj + self.ptt
##            self.psg = self.pzz + self.pzh

        self.pbgbg = self.pbg[self.y_true==bg.index]
        self.pbgsg = self.psg[self.y_true==bg.index]
        self.psgsg = self.psg[self.y_true==sg.index]
        self.psgbg = self.pbg[self.y_true==sg.index]

##            self.psgzz = self.y_pred[(self.y_true==zz.index)|(self.y_true==zh.index)][:,zz.index]
##            self.psgzh = self.y_pred[(self.y_true==zz.index)|(self.y_true==zh.index)][:,zh.index]
##            self.psgtt = self.y_pred[(self.y_true==zz.index)|(self.y_true==zh.index)][:,tt.index]
##            self.psgmj = self.y_pred[(self.y_true==zz.index)|(self.y_true==zh.index)][:,mj.index]
##
##            self.pbgzz = self.y_pred[(self.y_true==tt.index)|(self.y_true==mj.index)][:,zz.index]
##            self.pbgzh = self.y_pred[(self.y_true==tt.index)|(self.y_true==mj.index)][:,zh.index]
##            self.pbgtt = self.y_pred[(self.y_true==tt.index)|(self.y_true==mj.index)][:,tt.index]
##            self.pbgmj = self.y_pred[(self.y_true==tt.index)|(self.y_true==mj.index)][:,mj.index]

##            #regressed probabilities for ZZ to be each class
##            self.pzzzz = self.y_pred[self.y_true==zz.index][:,zz.index]
##            self.pzzzh = self.y_pred[self.y_true==zz.index][:,zh.index]
##            self.pzztt = self.y_pred[self.y_true==zz.index][:,tt.index]
##            self.pzzmj = self.y_pred[self.y_true==zz.index][:,mj.index]
##            self.pzzsg = self.psg[self.y_true==zz.index]
##            self.pzzbg = self.pbg[self.y_true==zz.index]
##
##            #regressed probabilities for ZH to be each class
##            self.pzhzz = self.y_pred[self.y_true==zh.index][:,zz.index]
##            self.pzhzh = self.y_pred[self.y_true==zh.index][:,zh.index]
##            self.pzhtt = self.y_pred[self.y_true==zh.index][:,tt.index]
##            self.pzhmj = self.y_pred[self.y_true==zh.index][:,mj.index]
##            self.pzhsg = self.psg[self.y_true==zh.index]
##            self.pzhbg = self.pbg[self.y_true==zh.index]
##
##            #regressed probabilities for ttbar to be each class
##            self.pttzz = self.y_pred[self.y_true==tt.index][:,zz.index]
##            self.pttzh = self.y_pred[self.y_true==tt.index][:,zh.index]
##            self.ptttt = self.y_pred[self.y_true==tt.index][:,tt.index]
##            self.pttmj = self.y_pred[self.y_true==tt.index][:,mj.index]
##            self.pttsg = self.psg[self.y_true==tt.index]
##            self.pttbg = self.pbg[self.y_true==tt.index]
##
##            #regressed probabilities for multijet model to be each class
##            self.pmjzz = self.y_pred[self.y_true==mj.index][:,zz.index]
##            self.pmjzh = self.y_pred[self.y_true==mj.index][:,zh.index]
##            self.pmjtt = self.y_pred[self.y_true==mj.index][:,tt.index]
##            self.pmjmj = self.y_pred[self.y_true==mj.index][:,mj.index]
##            self.pmjsg = self.psg[self.y_true==mj.index]
##            self.pmjbg = self.pbg[self.y_true==mj.index]

    def update(self, y_pred, y_true, q_score, w_ordered, loss, doROC=False, doBamber=False):
        self.y_pred = y_pred
        self.y_true = y_true
        self.q_score = q_score
        self.w      = w_ordered
        self.loss   = loss
        self.loss_min = loss if loss < self.loss_min else self.loss_min
        self.w_sum  = self.w.sum()

        self.q_1234 = self.q_score[:,0]
        self.q_1324 = self.q_score[:,1]
        self.q_1423 = self.q_score[:,2]

        # Weights for each class
        self.wbg = self.w[self.y_true==bg.index]
        self.wsg = self.w[self.y_true==sg.index]
##            self.wzz = self.w[self.y_true==zz.index]
##            self.wzh = self.w[self.y_true==zh.index]
##            self.wtt = self.w[self.y_true==tt.index]
##            self.wmj = self.w[self.y_true==mj.index]

        self.splitAndScale()

        if doROC:
            self.roc = roc_data(np.array(self.y_true==sg.index, dtype=np.float), 
                                self.y_pred[:,sg.index], 
                                self.w,
                                'Signal',
                                'Background')

            if doBamber:
#                self.roc.auc_ci = auc_ci.bamber(np.array(self.y_true==d4.index), self.y_pred[:,d4.index], self.roc.auc)
                self.roc_43.auc_ci = auc_ci.bootstrap(np.array(self.y_true==d4.index), self.y_pred[:,d4.index], self.roc_43.auc, self.w)
                print("Bamber CI: ", self.roc.auc_ci)

##                zhIndex = self.y_true!=zz.index
##                self.roc_zh = roc_data(np.array(self.y_true[zhIndex]==zh.index, dtype=np.float), 
##                                       self.y_pred[zhIndex][:,zh.index], 
##                                       self.w[zhIndex],
##                                       '$ZH$',
##                                       'Background')
##                zzIndex = self.y_true!=zh.index
##                self.roc_zz = roc_data(np.array(self.y_true[zzIndex]==zz.index, dtype=np.float), 
##                                       self.y_pred[zzIndex][:,zz.index], 
##                                       self.w[zzIndex],
##                                       '$ZZ$',
##                                       'Background')
##
    #binary signal vs background
    # class loaderResults:
    #     def __init__(self, name):
    #         self.name = name
    #         self.trainLoader = None
    #         self. evalLoader = None
    #         self.smallBatchLoader = None
    #         self.largeBatchLoader = None
    #         self.y_true = None
    #         self.y_pred = None
    #         self.n      = None
    #         self.w      = None
    #         self.roc= None #[0 for cl in classes]
    #         self.loss = 1e6
    #         self.loss_prev = None
    #         self.loss_best = 1e6
    #         self.roc_auc_best = None
    #         self.sum_w_S = None
    #         self.norm_d4_over_B = 0

    #     def splitAndScale(self):
    #         self.pbg = self.y_pred[:,bg.index]
    #         self.psg = self.y_pred[:,sg.index]

    #         self.pbgbg = self.pbg[(self.y_true==bg.index)]
    #         self.pbgsg = self.psg[(self.y_true==bg.index)]
    #         self.psgsg = self.psg[(self.y_true==sg.index)]
    #         self.psgbg = self.pbg[(self.y_true==sg.index)]

    #     def update(self, y_pred, y_true, w_ordered, loss, doROC=False):
    #         self.y_pred = y_pred
    #         self.y_true = y_true
    #         self.w      = w_ordered
    #         self.loss   = loss
    #         self.w_sum  = self.w.sum()

    #         # Weights for each class
    #         self.wbg  = self.w[(self.y_true==bg.index)]
    #         self.wsg  = self.w[(self.y_true==sg.index)]

    #         self.splitAndScale()

    #         if doROC:
    #             self.roc = roc_data(np.array((self.y_true==sg.index), dtype=np.float), 
    #                                 self.y_pred[:,sg.index], 
    #                                 self.w,
    #                                 'Signal',
    #                                 'Background')


class loaderResults_FvT:
    def __init__(self, name):
        self.name = name
        self.trainLoaders= []
        self.trainLoader = None
        self. evalLoader = None
        self.smallBatchLoader = None
        self.largeBatchLoader = None
        self.y_true = None
        self.y_pred = None
        self.n      = None
        self.w      = None
        self.roc= None #[0 for cl in classes]
        self.loss = 1e6
        self.loss_min = 1e6
        self.loss_prev = None
        self.loss_best = 1e6
        self.roc_auc_best = None
        self.sum_w_S = None
        self.probNorm_StoB = None
        self.probNorm_BtoS = None
        self.probNormRatio_StoB = None
        self.norm_d4_over_B = None

    def splitAndScale(self):
        self.pd3 = self.y_pred[:,d3.index]
##            self.pt3 = self.y_pred[:,t3.index]
        self.pd4 = self.y_pred[:,d4.index]
##            self.pt4 = self.y_pred[:,t4.index]

        #renormalize regressed probabilities such that their mean is as expected from the relative fraction of the samples
        self.pd3_ave     = (self.pd3 * self.w).sum()/self.w_sum
        self.pd3_ave_exp = self.wd3.sum()/self.w_sum
        self.sd3 = self.pd3_ave_exp/self.pd3_ave
##          self.pt3_ave     = (self.pt3 * self.w).sum()/self.w_sum
##            self.pt3_ave_exp = self.wt3.sum()/self.w_sum
##            self.st3 = self.pt3_ave_exp/self.pt3_ave
        self.pd4_ave     = (self.pd4 * self.w).sum()/self.w_sum
        self.pd4_ave_exp = self.wd4.sum()/self.w_sum
        self.sd4 = self.pd4_ave_exp/self.pd4_ave
##            self.pt4_ave     = (self.pt4 * self.w).sum()/self.w_sum
##            self.pt4_ave_exp = self.wt4.sum()/self.w_sum
##            self.st4 = self.pt4_ave_exp/self.pt4_ave

        # Compute multijet probabilities
        self.pm4 = self.pd4 ##- self.pt4
        self.pm3 = self.pd3 ##- self.pt3

        self.p4 = self.pd4 ##+ self.pt4
        self.p3 = self.pd3 ##+ self.pt3
        self.pd = self.pd4 ##+ self.pd3
##            self.pt = self.pt4 + self.pt3

        # # fix divide by zero
        # self.pm4[self.pm4>=0] += eps
        # self.pm4[self.pm4< 0] -= eps
        # self.pm3[self.pm3>=0] += eps
        # self.pm3[self.pm3< 0] -= eps

        # Compute reweight factor
        # self.r = self.pm4/self.pm3
        self.r = self.pm4/self.pd3

        #regressed probabilities for fourTag data to be each class
        self.pd4d4 = self.y_pred[self.y_true==d4.index][:,d4.index]
##            self.pd4t4 = self.y_pred[self.y_true==d4.index][:,t4.index]
        self.pd4d3 = self.y_pred[self.y_true==d4.index][:,d3.index]
##            self.pd4t3 = self.y_pred[self.y_true==d4.index][:,t3.index]

        #regressed probabilities for threeTag data to be each class
        self.pd3d4 = self.y_pred[self.y_true==d3.index][:,d4.index]
##           self.pd3t4 = self.y_pred[self.y_true==d3.index][:,t4.index]
        self.pd3d3 = self.y_pred[self.y_true==d3.index][:,d3.index]
##            self.pd3t3 = self.y_pred[self.y_true==d3.index][:,t3.index]

        #regressed probabilities for fourTag ttbar MC to be each class
##            self.pt4d4 = self.y_pred[self.y_true==t4.index][:,d4.index]
##            self.pt4t4 = self.y_pred[self.y_true==t4.index][:,t4.index]
##            self.pt4d3 = self.y_pred[self.y_true==t4.index][:,d3.index]
##            self.pt4t3 = self.y_pred[self.y_true==t4.index][:,t3.index]

        #regressed probabilities for threeTag ttbar MC to be each class
##            self.pt3d4 = self.y_pred[self.y_true==t3.index][:,d4.index]
##            self.pt3t4 = self.y_pred[self.y_true==t3.index][:,t4.index]
##            self.pt3d3 = self.y_pred[self.y_true==t3.index][:,d3.index]
##            self.pt3t3 = self.y_pred[self.y_true==t3.index][:,t3.index]


        #Define regressed probabilities for each class to be multijet
        self.pd4m4 = self.pm4[self.y_true==d4.index] # self.pd4d4 - self.pd4t4
        self.pd4m3 = self.pm3[self.y_true==d4.index] # self.pd4d3 - self.pd4t3
        self.pd3m4 = self.pm4[self.y_true==d3.index] # self.pd3d4 - self.pd3t4
        self.pd3m3 = self.pm3[self.y_true==d3.index] # self.pd3d3 - self.pd3t3
##            self.pt3m4 = self.pm4[self.y_true==t3.index] # self.pt3d4 - self.pt3t4
##            self.pt3m3 = self.pm3[self.y_true==t3.index] # self.pt3d3 - self.pt3t3
##            self.pt4m4 = self.pm4[self.y_true==t4.index] # self.pt4d4 - self.pt4t4
##            self.pt4m3 = self.pm3[self.y_true==t4.index] # self.pt4d3 - self.pt4t3

        #Compute multijet weights for each class
        self.rd4 = self.r[self.y_true==d4.index] # self.pd4m4/self.pd4m3
        self.rd3 = self.r[self.y_true==d3.index] # self.pd3m4/self.pd3m3
##            self.rt4 = self.r[self.y_true==t4.index] # self.pt4m4/self.pt4m3
##            self.rt3 = self.r[self.y_true==t3.index] # self.pt3m4/self.pt3m3

        #Compute normalization of the reweighted background model
        # self.normB = ( self.wd3 * self.rd3 ).sum() - ( self.wt3 * self.rt3 ).sum() + self.wt4.sum()
        self.normB = ( self.wd3 * self.rd3 ).sum() ##+ self.wt4.sum()
        self.norm_d4_over_B = self.wd4.sum()/self.normB

    def update(self, y_pred, y_true, q_score, w_ordered, loss, doROC=False, doBamber=False):
        self.y_pred = y_pred
        self.y_true = y_true
        self.q_score =  q_score
        self.w      = w_ordered
        self.loss   = loss
        self.loss_min = loss if loss < self.loss_min else self.loss_min
        self.w_sum  = self.w.sum()
        #self.wB = np.copy(self.w)
        #self.wB[self.y_true==t3.index] *= -1

        self.q_1234 = self.q_score[:,0]
        self.q_1324 = self.q_score[:,1]
        self.q_1423 = self.q_score[:,2]

        # Weights for each class
        self.wd4 = self.w[self.y_true==d4.index]
##            self.wt4 = self.w[self.y_true==t4.index]
        self.wd3 = self.w[self.y_true==d3.index]
##            self.wt3 = self.w[self.y_true==t3.index]

##            self.wt4n = self.wt4[self.wt4<0]
##            self.wt3n = self.wt3[self.wt3<0]
        #print()
        #print("t3 negative fraction:",abs(self.wt3n.sum())/self.wt3.sum())
        #print("t4 negative fraction:",abs(self.wt4n.sum())/self.wt4.sum())

        self.splitAndScale()

        ## TODO negative weight roc code.

        if doROC:
#            if classifier in ['DvT3']:
#                self.roc_t3 = roc_data(np.array(self.y_true==t3.index, dtype=np.float), 
#                                       self.y_pred[:,t3.index], 
#                                       self.w,
#                                       r'ThreeTag $t\bar{t}$ MC',
#                                       'ThreeTag Data')
#                self.roc = self.roc_t3
#            if classifier in ['FvT','DvT4']:
            self.roc_43 = roc_data(np.array(self.y_true==d4.index, dtype=np.float), 
                                   self.y_pred[:,d4.index], 
                                   self.w,
                                   'FourTag',
                                   'ThreeTag')
## TODO
            if doBamber:
                #self.roc_43.auc_ci = auc_ci.bamber(np.array(self.y_true==d4.index), self.y_pred[:,d4.index], self.roc_43.auc)
                self.roc_43.auc_ci = auc_ci.bootstrap(np.array(self.y_true==d4.index), self.y_pred[:,d4.index], self.roc_43.auc, self.w)

                print("Bamber CI: ", self.roc_43.auc_ci)
            self.roc = self.roc_43 #+ self.roc_td.auc - 1

class modelParameters:

##    def __init__(self, dfB, dfS, yTrueLabel='fourTag', filename=None, classifier='FvT', weight='weight',
##                 device=None, n_queue=20, eval_batch_size=16384, print_step=10, rate_StoS=None, rate_BtoB=None, barScale=750, barMin=0.5,
##                 train_fraction=0.7, train_batch_size_small=64, doROC=True, nodes=32, layers=3, pDropout=0.4, epochs=20, lrInit=1e-3, use_mass=False,
##                 signalName="4b", backgroundName="3b", bamber=True):



    def __init__(self, dfB, dfS, lrInit=0.01, train_batch_size=512, num_params=6, yTrueLabel='fourTag', fileName=None, classifier='FvT', bamber=True, epochs=10, fixedSchedule=True, bs_milestones=[1,3,6], lr_milestones=[8,9],
                 model_path=False, onnx=False, update=False, storeEventFile=False):

        self.modelPath = model_path
        
        self.fileName = fileName
        self.bamber = bamber
        self.train_batch_size = train_batch_size
        self.fixedSchedule = fixedSchedule
        self.bs_milestones = bs_milestones
        self.lr_milestones = lr_milestones

        self.classifier = classifier

        train_numerator = 2
        train_denominator = 3
        train_fraction = train_numerator/train_denominator
        train_offset = 0

        if classifier in ['SvB']:
            #eval_batch_size = 2**15
            train_batch_size = 2**10
            lrInit = 0.8e-2
        
            self.barMin=0.85
            self.barScale=1000
            self.weight = 'weight'

            ##yTrueLabel = 'target'
            self.yTrueLabel = 'signal'

            ##classes = [zz,zh,tt,mj]
            eps = 0.0001
        
            #nClasses = len(classes)
        
            ## Uncommented by T
            classes = [sg,bg]
            nClasses = 2
            self.nClasses=2     
            self.classes=classes
            #wC = torch.FloatTensor([1 for i in range(nClasses)]).to("cuda")

            dfB[self.yTrueLabel] = bg.index 
            dfS[self.yTrueLabel] = sg.index
             
        
            updateAttributes = [
                nameTitle('pzz', classifier+'_pzz'),
                nameTitle('pzh', classifier+'_pzh'),
                nameTitle('ptt', classifier+'_ptt'),
                nameTitle('pmj', classifier+'_pmj'),
                nameTitle('psg', classifier+'_ps'),
                nameTitle('pbg', classifier+'_pb'),
                nameTitle('q_1234', classifier+'_q_1234'),
                nameTitle('q_1324', classifier+'_q_1324'),
                nameTitle('q_1423', classifier+'_q_1423'),
                ]
        
            if not update and not storeEventFile and not onnx:
                train_fraction = 0.7
        
                # Read .h5 files
                ##results = fileReaders.map_async(getFrameSvB, sorted(glob(args.data)))
                ##frames = results.get()
                ##dfDB = pd.concat(frames, sort=False)
                ##dfDB[weight] = dfDB['mcPseudoTagWeight'] * dfDB['FvT']
                ##nDB = dfDB.shape[0]
                ##wDB = np.sum( dfDB[weight] )
                ##print("nDB",nDB)
                ##print("wDB",wDB)
        
                ##results = fileReaders.map_async(getFrameSvB, sorted(glob(args.ttbar)))
                ##frames = results.get()
                ##dfT = pd.concat(frames, sort=False)
        
                ##nT = dfT.shape[0]
                ##wT = np.sum(dfT[weight])
                ##print("nT",nT)
                ##print("wT",wT)
        
                ##dfB = pd.concat([dfDB, dfT], sort=False)
                ##    
                ##results = fileReaders.map_async(getFrameSvB, sorted(glob(args.signal)))
                ##frames = results.get()
                ##dfS = pd.concat(frames, sort=False)
        
                nS      = dfS.shape[0]
                nB      = dfB.shape[0]
                print("nS",nS)
                print("nB",nB)
        
                # compute relative weighting for S and B
                ##nzz, wzz = dfS.zz.sum(), dfS.loc[dfS.zz==1][weight].sum()
                ##nzh, wzh = dfS.zh.sum(), dfS.loc[dfS.zh==1][weight].sum()
                sum_wS = np.sum(np.float32(dfS[self.weight]))
                sum_wB = np.sum(np.float32(dfB[self.weight]))
                print("sum_wS",sum_wS)
                print("sum_wB",sum_wB)
                ##print("nzz = %7d, wzz = %6.1f"%(nzz,wzz))
                ##print("nzh = %7d, wzh = %6.1f"%(nzh,wzh))

                ## TODO normalize sig background        
                dfS[self.weight] *= sum_wB/sum_wS #normalize signal to background
                ##dfS[weight] = dfS[weight]*(dfS.zz==1)*sum_wB/wzz + dfS[weight]*(dfS.zh==1)*sum_wB/wzh
                #dfS[weight] = dfS[weight]*(dfS.zh==1)*sum_wB/wzh
    

                ###
                ### Split into training and validation sets
                ###
                ##nTrainS = int(nS*train_fraction)
                ##nTrainB = int(nB*train_fraction)
                ##nValS   = nS-nTrainS
                ##nValB   = nB-nTrainB
        
                ###random ordering to mix up which data is used for training or validation
                ##idxS    = np.random.permutation(nS)
                ##idxB    = np.random.permutation(nB)
        
                ###define dataframes for trainging and validation
                ##dfS_train = dfS.iloc[idxS[:nTrainS]]
                ##dfS_val   = dfS.iloc[idxS[nTrainS:]]
                ##dfB_train = dfB.iloc[idxB[:nTrainB]]
                ##dfB_val   = dfB.iloc[idxB[nTrainB:]]
            
                ##df_train = pd.concat([dfB_train, dfS_train], sort=False)
                ##df_val   = pd.concat([dfB_val,   dfS_val  ], sort=False)

                ##self.df_train = df_train
                ##self.df_val = df_val

                df = pd.concat([dfB, dfS], sort=False)
                n = df.shape[0]
                #
                # Split into training and validation sets
                #
                idx_train, idx_val = [], []
                print("build idx with offset %i, modulus %i, and train/val split %i"%(train_offset, train_denominator, train_numerator))
                for e in range(n):
                    if (e+train_offset)%train_denominator < train_numerator: 
                        idx_train.append(e)
                    else:
                        idx_val  .append(e)
                idx_train, idx_val = np.array(idx_train), np.array(idx_val)

                print("Split into training and validation sets")
                self.df_train, self.df_val = df.iloc[idx_train], df.iloc[idx_val]

        #self.barMin, self.barScale = 0.5, 200

        if classifier in ['FvT']:
            self.yTrueLabel = 'fourTag'


            self.barMin, self.barScale = 0.57, 1000
            self.weight = 'mcPseudoTagWeight' #### TODO
#            yTrueLabel = 'target'
        
            self.classes = [d4,d3]##,t4,t3]
            eps = 0.0001
        
            self.nClasses = len(self.classes)
        
            if classifier in ['FvT', 'DvT3', 'DvT4']: 
                updateAttributes = [
                    nameTitle('r',   classifier),
                    nameTitle('pd4', classifier+'_pd4'),
                    nameTitle('pd3', classifier+'_pd3'),
                    nameTitle('pt4', classifier+'_pt4'),
                    nameTitle('pt3', classifier+'_pt3'),
                    nameTitle('pm4', classifier+'_pm4'),
                    nameTitle('pm3', classifier+'_pm3'),
                    nameTitle('p4',  classifier+'_p4'),
                    nameTitle('p3',  classifier+'_p3'),
                    nameTitle('pd',  classifier+'_pd'),
                    nameTitle('pt',  classifier+'_pt'),
                    nameTitle('q_1234', classifier+'_q_1234'),
                    nameTitle('q_1324', classifier+'_q_1324'),
                    nameTitle('q_1423', classifier+'_q_1423'),
                    ]
        
            if not update_global and not storeEventFile and not onnx:
        
                # Read .h5 files
        ##        results = fileReaders.map_async(getFrame, sorted(glob(args.data)))
        ##        frames = results.get()
        ##        dfD = pd.concat(frames, sort=False)
        ##        dfD = pd.concat([df3b, df4b], sort=False) mine.
                dfD = pd.concat([dfB, dfS], sort=False)
        
                print("Add true class labels to data")
                dfD['d4'] =  dfD.fourTag
                dfD['d3'] = (dfD.fourTag+1)%2
        
                print("concatenate data and ttbar dataframes")
        ##        df = pd.concat([dfD, dfT], sort=False)
        ##        df = dfD
        
                df = dfD

                ## Tudor --- setting these all to 1 for the toy data.
                df['pseudoTagWeight'] = 1
                df['mcPseudoTagWeight'] = df['weight']
                ##df['xW'] = 1
                ##df['xbW']=1
                ##df['nSelJets'] = 1
                ##df['year'] = 1
                ##df['xWt'] = 1
                ##df['nSelJets'] = 1

                ####print("____ DF SHAPE ______")
                ####print(df.shape[0])
        
                ####df.drop(df[df['weight'] == 0].index, inplace=True)

                ####df.to_hdf("horizout.h5", key="df", mode="w") 
                ####n = df.shape[0]
                ####print("EFFN ", n)


                print("add encoded target")
                df['target'] = d4.index*df.d4 + d3.index*df.d3 ##+ t4.index*df.t4 + t3.index*df.t3 
                # classes are mutually exclusive so the target computed in this way is 0,1,2 or 3.
        
                n = df.shape[0]
        
                nd4, wd4 = df.d4.sum(), df.loc[df.d4==1].mcPseudoTagWeight.sum()
                nd3, wd3 = df.d3.sum(), df.loc[df.d3==1].mcPseudoTagWeight.sum()
        
                w = wd4+wd3##+wt4+wt3
        
                fC =      torch.FloatTensor([wd4/w, wd3/w])##, wt4/w, wt3/w])
        
                print("nd4 = %7d, wd4 = %6.1f"%(nd4,wd4))
                print("nd3 = %7d, wd3 = %6.1f"%(nd3,wd3))
        ##        print("nt4 = %7d, wt4 = %6.1f"%(nt4,wt4))
        ##        print("nt3 = %7d, wt3 = %6.1f"%(nt3,wt3))
        ##        print("wtn = %6.1f"%(wtn))
                print("fC:",fC)
                ##print("wC:",wC)
        ### END IF
        
                #
                # Split into training and validation sets
                #
                idx_train, idx_val = [], []
                print("build idx with offset %i, modulus %i, and train/val split %i"%(train_offset, train_denominator, train_numerator))
                for e in range(n):
                    if (e+train_offset)%train_denominator < train_numerator: 
                        idx_train.append(e)
                    else:
                        idx_val  .append(e)
                idx_train, idx_val = np.array(idx_train), np.array(idx_val)

                print("Split into training and validation sets")
                self.df_train, self.df_val = df.iloc[idx_train], df.iloc[idx_val]


        self.xVariables=['canJet0_pt', 'canJet1_pt', 'canJet2_pt', 'canJet3_pt',
                         'dRjjClose', 'dRjjOther', 
                         'aveAbsEta', 
                          'xWt',    ##
                         'nSelJets',## 
                         'm4j',
                         ]



        #             |1|2|3|4|1|3|2|4|1|4|2|3|  ##stride=2 kernel=2 gives all possible dijets
        self.layer1Pix = "012302130312"
        #self.canJets=[['canJet%s_pt'%i, 'canJet%s_eta'%i, 'canJet%s_phi'%i, 'canJet%s_m'%i] for i in self.layer1Pix] #index[pixel][color]
        self.canJets = ['canJet%s_pt' %i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_eta'%i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_phi'%i for i in self.layer1Pix]
## TODO could add an argument for whether or not to use mass here.
##        self.canJets+= ['canJet%s_m'  %i for i in self.layer1Pix]
        self.canJets+= ['canJet%s_m'  %i for i in self.layer1Pix]
        
        #self.canJetMean = [110, 0.0,  0.00, 15]
        #self.canJetStd  = [ 40, 2.4, np.pi,  5]
        self.canJetMean = [0,0, 0.00,0]
        self.canJetStd  = [1,1,np.pi,1]

        self.useOthJets = ''
        self.othJetMean = [0,0, 0.00,0,0]
        self.othJetStd  = [1,1,np.pi,1,1]

        self.jetFeatures = 4
        self.othJetFeatures = 5

        self.dijetAncillaryFeatures=[ 'm01',  'm23',  'm02',  'm13',  'm03',  'm12',
                                     'dR01', 'dR23', 'dR02', 'dR13', 'dR03', 'dR12',
                                    #'pt01', 'pt23', 'pt02', 'pt13', 'pt03', 'pt12',
                                      ]
        #self.dijetMean = [130, np.pi/2]
        #self.dijetStd  = [100, np.pi/2]e
        self.dijetMean = [0,0]
        self.dijetStd  = [1,1]
        self.nAd = 2

        self.quadjetAncillaryFeatures=['dR0123', 'dR0213', 'dR0312',
                                       'm4j',    'm4j',    'm4j',
                                       ##'xW',     'xW',     'xW',           
                                       ##'xbW',    'xbW',    'xbW',          
                                       ##'nSelJets', 'nSelJets', 'nSelJets', 
                                       ##'year',   'year',   'year',                                       
                                       ]                                  
        #self.quadjetMean = [np.pi  , 500, 0, 0, 7, 2017]                 
        #self.quadjetStd  = [np.pi/2, 200, 5, 5, 3,    3]
        self.quadjetMean = [0,0]##,0,0,0,0]
        self.quadjetStd  = [1,1]##,1,1,1,1]
        self.nAq = 2##6

        self.trainingHistory = {}

        if classifier == 'SvB':
            self.validation = loaderResults_SvB("validation")
            self.training   = loaderResults_SvB("training")

        else:
            self.validation = loaderResults_FvT("validation")
            self.training   = loaderResults_FvT("training")

        self.train_losses = []
        self.valid_losses = []
        self.train_aucs = []
        self.valid_aucs = []
        self.train_stats = []
        self.valid_stats = []
        self.bs_change = []
        self.lr_change = []

        lossDict = {'FvT': 0.3,#0.1485,
                    'DvT3': 0.065,
                    'ZZvB': 1,
                    'ZHvB': 1,
                    'SvB': 0.2120,
                    }
        
        if fileName:
            self.classifier           = classifier #    fileName.split('_')[0] if not 
            if "FC" in fileName:
                self.nodes         =   int(fileName[fileName.find(      'x')+1 : fileName.find('_pdrop')])
                self.layers        =   int(fileName[fileName.find(     'FC')+2 : fileName.find('x')])
                self.dijetFeatures = None
                self.quadjetFeatures = None
                self.combinatoricFeatures = None
                self.pDropout      = float(fileName[fileName.find( '_pdrop')+6 : fileName.find('_np')])

            ## TODO set these to 6?
            if "ResNet" in fileName:
                self.dijetFeatures        = num_params #6 ##int(fileName.split('_')[2])
                self.quadjetFeatures      = num_params #6 ##int(fileName.split('_')[3])
                self.combinatoricFeatures = num_params #6 ##int(fileName.split('_')[4])
                self.nodes    = None
                self.pDropout = None
            ## TODO
            self.lrInit        =       float(fileName[fileName.find(    '_lr')+3 : fileName.find('_epochs')])
            self.startingEpoch =       int(fileName[fileName.find('e_epoch')+7 : fileName.find('_loss')])
            self.training.loss_best  = float(fileName[fileName.find(   '_loss')+5 : fileName.find('.pkl')])

        else:
            self.dijetFeatures  = num_params# 9
            self.quadjetFeatures = num_params#9
            self.combinatoricFeatures = num_params#9
            self.nodes         = nodes
            self.layers        = layers
            self.pDropout      = pDropout
            self.lrInit        = lrInit
            self.startingEpoch = 0           
            self.training.loss_best  = lossDict[self.classifier]
            if self.classifier in ['M1vM2']: self.validation.roc_auc_best = 0.5

        print("--------------- LRINIT: ", self.lrInit)
        
        self.modelPkl = fileName
        self.epoch = self.startingEpoch

        print(device)
        self.net = ResNet(self.jetFeatures, self.dijetFeatures, self.quadjetFeatures, self.combinatoricFeatures, self.useOthJets, device=device, nClasses=self.nClasses).to(device)
        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.epochs = epochs
        self.lastEpoch = self.epochs + self.startingEpoch
        self.name = outputName+self.classifier+'_'+self.net.name+'_np%d_lr%s_epochs%d_stdscale'%(self.nTrainableParameters, str(self.lrInit), self.lastEpoch)
        ## TODO 
        ##self.logFileName = 'ZZ4b/nTupleAnalysis/pytorchModels/'+self.name+'.log'
        self.logFileName = 'log_file.log'
        print("Set log file:", self.logFileName)
        self.logFile = open(self.logFileName, 'a', 1)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lrInit, amsgrad=False)
        #self.optimizer = NAdam(self.net.parameters(), lr=self.lrInit)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=0.8, momentum=0.9, nesterov=True)
        self.patience = 0
        self.max_patience = max_patience
        if self.fixedSchedule:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_milestones, gamma=0.1, last_epoch=-1)
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, threshold=0, patience=self.max_patience, cooldown=1, min_lr=2e-4, verbose=True)
                
        self.foundNewBest = False
        
        self.dump()

        if fileName:
            print("Load Model:", fileName)
            self.net.load_state_dict(torch.load(fileName)['model']) # load model from previous state
            self.optimizer.load_state_dict(torch.load(fileName)['optimizer'])
            self.trainingHistory = torch.load(fileName)['training history']

            if onnx:
                self.exportONNX()
                exit()

            if storeEventFile:
                files = []
                for sample in [args.data, args.ttbar, args.signal]:
                    files += sorted(glob(sample))
                self.storeEvent(files[0], args.storeEvent)
                exit()
            global_update = False ###
            if global_update:
                files = []
                for sample in [args.data, args.ttbar, args.signal]:
                    files += sorted(glob(sample))
                for sampleFile in files:
                    print(sampleFile)
                for sampleFile in files:
                    self.update(sampleFile)
                exit()

    def logprint(self, s, end='\n'):
        print(s,end=end)
        self.logFile.write(s+end)

    def epochString(self):
        return ('>> %'+str(len(str(self.lastEpoch)))+'d/%d <<')%(self.epoch, self.lastEpoch)

    def dfToTensors(self, df, y_true=None):
        n = df.shape[0]
        #basicDNN variables
        #X=torch.FloatTensor( np.float32(df[self.xVariables]) )

        #jet features
        J=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.canJets], 1 )
##        O=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.othJets], 1 )
        O = np.array([])

        #extra features 
        D=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.dijetAncillaryFeatures], 1 )
        Q=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.quadjetAncillaryFeatures], 1 )
        #A=torch.cat( [torch.FloatTensor( np.float32(df[feature]).reshape(-1,1) ) for feature in self.ancillaryFeatures], 1 ) 

        if y_true:
            y=torch.LongTensor( np.array(df[y_true], dtype=np.uint8).reshape(-1) )
        else:#assume all zero. y_true not needed for updating classifier output values in .h5 files for example.
            y=torch.LongTensor( np.zeros(df.shape[0], dtype=np.uint8).reshape(-1) )

        w=torch.FloatTensor( np.float32(df[self.weight]).reshape(-1) )

        return J, O, D, Q, y, w


    def storeEvent(self, fileName, eventRow):
        print("Store network response for", self.classifier,"from file",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
        yearIndex = fileName.find('201')
        year = float(fileName[yearIndex:yearIndex+4])
        print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)

        print("Grab event from row",eventRow)
        i = pd.RangeIndex(df.shape[0])
        df.set_index(i, inplace=True)
        df = df.iloc[int(eventRow):int(eventRow)+1,:]
        print(df)

        n = df.shape[0]
        print("Convert df to tensors",n)

        J, O, D, Q, y, w = self.dfToTensors(df)

        # Set up data loaders
        print("Make data loader")
        dset   = TensorDataset(J, O, D, Q, y, w)
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        updateResults.n = n

        self.net.store=storeEventFile

        self.evaluate(updateResults, doROC = False)

        self.net.writeStore()


    def update(self, fileName):
        print("Add",self.classifier,"output to",fileName)
        # Read .h5 file
        df = pd.read_hdf(fileName, key='df')
        yearIndex = fileName.find('201')
        year = float(fileName[yearIndex:yearIndex+4])
        print("Add year to dataframe",year)#,"encoded as",(year-2016)/2)
        df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)

        n = df.shape[0]
        print("Convert df to tensors",n)

        J, O, D, Q, y, w = self.dfToTensors(df)

        # Set up data loaders
        print("Make data loader")
        dset   = TensorDataset(J, O, D, Q, y, w)
        updateResults = loaderResults("update")
        updateResults.evalLoader = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        updateResults.n = n

        self.evaluate(updateResults, doROC = False)

        for attribute in updateAttributes:
            df[attribute.title] = pd.Series(np.float32(getattr(updateResults, attribute.name)), index=df.index)

        df.to_hdf(fileName, key='df', format='table', mode='w')
        del df
        del dset
        del updateResults
        print("Done")

    def exportONNX(self):
        self.net.onnx = True # apply softmax to class scores
        # Create a random input for the network. The onnx export will use this to trace out all the operations done by the model.
        # We can later check that the model output is the same with onnx and pytorch evaluation.
        # test_input = (torch.ones(1, 4, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, 5, 12, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAd, 6, requires_grad=True).to('cuda'),
        #               torch.ones(1, self.net.nAq, 3, requires_grad=True).to('cuda'),
        #               )
        J = torch.tensor([182.747, 141.376, 109.942, 50.8254, 182.747, 109.942, 141.376, 50.8254, 182.747, 50.8254, 141.376, 109.942, 
                          0.772827, 1.2832, 1.44385, 2.06543, 0.772827, 1.44385, 1.2832, 2.06543, 0.772827, 2.06543, 1.2832, 1.44385, 
                          2.99951, -0.797241, 0.561157, -2.83203, 2.99951, 0.561157, -0.797241, -2.83203, 2.99951, -2.83203, -0.797241, 0.561157, 
                          14.3246, 10.5783, 13.1129, 7.70751, 14.3246, 13.1129, 10.5783, 7.70751, 14.3246, 7.70751, 10.5783, 13.1129],
                         requires_grad=True).to('cuda').view(1,48)
##        O = torch.tensor([22.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##                          0.0322418, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
##                          -0.00404358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##                          4.01562, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
##                          -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
##                         requires_grad=True).to('cuda').view(1,60)
        O = np.array([])
        D = torch.tensor([316.5, 157.081, 284.569, 160.506, 142.039, 159.722, 
                          2.53827, 2.95609, 2.529, 2.17997, 1.36923, 1.36786],
                         requires_grad=True).to('cpu').view(1,12)
##                         requires_grad=True).to('cuda').view(1,12)
        Q = torch.tensor([3.18101, 2.74553, 2.99015, 
                          525.526, 525.526, 525.526,    ##],
                          4.51741, 4.51741, 4.51741,    ## 
                          0.554433, 0.554433, 0.554433, ##
                          4, 4, 4,                      ##
                          2016, 2016, 2016],            ##
                         requires_grad=True).to('cpu').view(1, 18)##view(1,18)
#                         requires_grad=True).to('cuda').view(1,18)
        # Export the model
        self.net.eval()
        torch_out = self.net(J, O, D, Q)
        print("test output:",torch_out)
        self.modelONNX = self.modelPkl.replace('.pkl','.onnx')
        print("Export ONNX:",self.modelONNX)
        torch.onnx.export(self.net,                                        # model being run
                          (J, O, D, Q),                                    # model input (or a tuple for multiple inputs)
                          self.modelONNX,                                  # where to save the model (can be a file or file-like object)
                          export_params=True,                              # store the trained parameter weights inside the model file
                          #opset_version= 7,                               # the ONNX version to export the model to
                          #do_constant_folding=True,                       # whether to execute constant folding for optimization
                          input_names  = ['J','O','D','Q'],                # the model's input names
                          output_names = ['c_score', 'q_score'],           # the model's output names
                          #dynamic_axes={ 'input' : {0 : 'batch_size'},    # variable lenght axes
                          #              'output' : {0 : 'batch_size'}}
                          verbose = False
                          )

        # import onnx
        # onnx_model = onnx.load(self.modelONNX)
        # # Check that the IR is well formed
        # onnx.checker.check_model(onnx_model)
        

    def trainSetup(self):
        df_train = self.df_train
        df_val = self.df_val
      
        print("Convert df_train to tensors")
        J_train, O_train, D_train, Q_train, y_train, w_train = self.dfToTensors(df_train, y_true=self.yTrueLabel)
        print("Convert df_val to tensors")
        J_val  , O_val  , D_val  , Q_val  , y_val  , w_val   = self.dfToTensors(df_val  , y_true=self.yTrueLabel)
        print('J_train.shape, O_train.shape, y_train.shape, w_train.shape:', J_train.shape, O_train.shape, y_train.shape, w_train.shape)
        print('J_val  .shape, O_val  .shape, y_val  .shape, w_val  .shape:', J_val  .shape, O_val  .shape, y_val  .shape, w_val  .shape)

        # Standardize inputs
        if not self.fileName: 
##            self.net.canJetScaler.m = torch.tensor(self.canJetMean, dtype=torch.float).view(1,-1,1).to('cuda')
##            self.net.canJetScaler.s = torch.tensor(self.canJetStd,  dtype=torch.float).view(1,-1,1).to('cuda')
##
##            if self.useOthJets:
##                self.net.othJetScaler.m = torch.tensor(self.othJetMean, dtype=torch.float).view(1,-1,1).to('cuda')
##                self.net.othJetScaler.s = torch.tensor(self.othJetStd,  dtype=torch.float).view(1,-1,1).to('cuda')
##
##            self.net.dijetScaler.m = torch.tensor(self.dijetMean, dtype=torch.float).view(1,-1,1).to('cuda')
##            self.net.dijetScaler.s = torch.tensor(self.dijetStd,  dtype=torch.float).view(1,-1,1).to('cuda')
##
##            self.net.quadjetScaler.m = torch.tensor(self.quadjetMean, dtype=torch.float).view(1,-1,1).to('cuda')
##            self.net.quadjetScaler.s = torch.tensor(self.quadjetStd , dtype=torch.float).view(1,-1,1).to('cuda')

            self.net.canJetScaler.m = torch.tensor(self.canJetMean, dtype=torch.float).view(1,-1,1).to('cpu')
            self.net.canJetScaler.s = torch.tensor(self.canJetStd,  dtype=torch.float).view(1,-1,1).to('cpu')

            if self.useOthJets:
                self.net.othJetScaler.m = torch.tensor(self.othJetMean, dtype=torch.float).view(1,-1,1).to('cpu')
                self.net.othJetScaler.s = torch.tensor(self.othJetStd,  dtype=torch.float).view(1,-1,1).to('cpu')

            self.net.dijetScaler.m = torch.tensor(self.dijetMean, dtype=torch.float).view(1,-1,1).to('cpu')
            self.net.dijetScaler.s = torch.tensor(self.dijetStd,  dtype=torch.float).view(1,-1,1).to('cpu')

            self.net.quadjetScaler.m = torch.tensor(self.quadjetMean, dtype=torch.float).view(1,-1,1).to('cpu')
            self.net.quadjetScaler.s = torch.tensor(self.quadjetStd , dtype=torch.float).view(1,-1,1).to('cpu')

        # Set up data loaders
        dset_train   = TensorDataset(J_train, D_train, D_train, Q_train, y_train, w_train)  ## TODO Important.
        dset_val     = TensorDataset(J_val,   D_val,   D_val,   Q_val,   y_val,   w_val)    ## TODO
        self.training.trainLoaders.append( DataLoader(dataset=dset_train, batch_size=self.train_batch_size*16, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True) )
        self.training.trainLoaders.append( DataLoader(dataset=dset_train, batch_size=self.train_batch_size*8, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True) )
        self.training.trainLoaders.append( DataLoader(dataset=dset_train, batch_size=self.train_batch_size*4, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True) )
        self.training.trainLoaders.append( DataLoader(dataset=dset_train, batch_size=self.train_batch_size*1, shuffle=True,  num_workers=n_queue, pin_memory=True, drop_last=True) )
        self.training  .evalLoader       = DataLoader(dataset=dset_train, batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.validation.evalLoader       = DataLoader(dataset=dset_val,   batch_size=eval_batch_size,        shuffle=False, num_workers=n_queue, pin_memory=True)
        self.training.n, self.validation.n = w_train.shape[0], w_val.shape[0]
        self.training .trainLoader     = self.training.trainLoaders.pop() # start with smallest batch size
        print("Training Batch Size:",self.train_batch_size)
        print("Training Batches:",len(self.training.trainLoader))

        #model initial state
        epochSpaces = max(len(str(self.epochs))-2, 0)
        stat = 'Norm' if self.classifier == 'FvT' else 'Sig.'
        self.logprint(">> "+(epochSpaces*" ")+"Epoch"+(epochSpaces*" ")+" <<   Data Set |  Loss  | "+stat+" | % AUC | AUC Bar Graph ^ Overtraining Metric * Output Model")
        self.trainEvaluate(doROC=True)
        self.validate(doROC=True)
        self.logprint('')
        if self.fixedSchedule:
            self.scheduler.step()
        else:
            self.scheduler.step(self.training.loss)


    def evaluate(self, results, doROC=True, evalOnly=False):
        self.net.eval()
        y_pred, y_true, w_ordered = np.ndarray((results.n,self.nClasses), dtype=np.float), np.zeros(results.n, dtype=np.float), np.zeros(results.n, dtype=np.float)
        q_score = np.ndarray((results.n, 3), dtype=np.float)
        print_step = len(results.evalLoader)//200+1
        nProcessed = 0
        loss = 0
        for i, (J, O, D, Q, y, w) in enumerate(results.evalLoader):
            nBatch = w.shape[0]
            J, O, D, Q, y, w = J.to(device), O.to(device), D.to(device), Q.to(device), y.to(device), w.to(device)
            logits, quadjet_scores = self.net(J, O, D, Q)
            loss += (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).sum(dim=0).cpu().item()
            y_pred[nProcessed:nProcessed+nBatch] = F.softmax(logits, dim=-1).detach().cpu().numpy()
            y_true[nProcessed:nProcessed+nBatch] = y.cpu()
            q_score[nProcessed:nProcessed+nBatch] = quadjet_scores.detach().cpu().numpy()
            w_ordered[nProcessed:nProcessed+nBatch] = w.cpu()

            nProcessed+=nBatch
            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(results.evalLoader)
                sys.stdout.write('\rEvaluating %3.0f%%     '%(percent))
                sys.stdout.flush()
                
        loss = loss/w_ordered.sum()
        print("Running update now")
        print("self epoch")
        print(self.epoch)
        results.update(y_pred, y_true, q_score, w_ordered, loss, doROC, doBamber=(self.epoch==10))


    def validate(self, doROC=True):
        self.evaluate(self.validation, doROC)
        bar=self.validation.roc.auc
        bar=int((bar-self.barMin)*self.barScale) if bar > self.barMin else 0

        # roc_abc=None
        overtrain=""
        if self.training.roc: 
            try:
                n = self.validation.roc.fpr.shape[0]
                roc_val = interpolate.interp1d(self.validation.roc.fpr[np.arange(0,n,n//100)], self.validation.roc.tpr[np.arange(0,n,n//100)], fill_value="extrapolate")
                tpr_val = roc_val(self.training.roc.fpr)#validation tpr estimated at training fpr
                n = self.training.roc.fpr.shape[0]
                roc_abc = auc(self.training.roc.fpr[np.arange(0,n,n//100)], np.abs(self.training.roc.tpr-tpr_val)[np.arange(0,n,n//100)]) #area between curves

                """
                bins  = [b/10.0 for b in range(0,501)]
                bins += [1e6] #overflow
                ce_hist_validation, _ = np.histogram(self.validation.cross_entropy*self.validation.w, bins=bins)#, weights=self.validation.w)
                ce_hist_training  , bins = np.histogram(self.training  .cross_entropy*self.training  .w, bins=bins)#, weights=self.training  .w)
                ce_hist_training = ce_hist_training * self.validation.n/self.training.n
                #remove bins where f_exp is less than ten for chisquare test (assumes gaussian rather than poisson stats). Use validation as f_obs and training as f_exp
                bins = bins[:-1]
                bins = bins[ce_hist_training>10]
                ce_hist_validation = ce_hist_validation[ce_hist_training>10]
                ce_hist_training   = ce_hist_training  [ce_hist_training>10]
                chi2 = chisquare(ce_hist_validation, ce_hist_training)
                ndf = len(ce_hist_validation)
                roc_percent = roc_abc*100 / (roc_abc + (self.training.roc.auc-0.5 if self.validation.roc.auc > 0.5 else 0))
                overtrain="^ %1.1f%%"%(roc_percent, bins[-1], ndf, chi2.statistic/ndf)  ##/(self.training.roc.auc-0.5))
                """
                roc_percent = roc_abc*100 / (roc_abc + (self.training.roc.auc-0.5 if self.training.roc.auc > 0.5 else 0))
                overtrain="^ %1.1f%%"%(roc_percent)  ##/(self.training.roc.auc-0.5))
            except ZeroDivisionError:
                overtrain="NaN"
        stat = self.validation.norm_d4_over_B if self.classifier == 'FvT' else self.validation.roc.maxSigma
        print('\r', end = '')
        s=self.epochString()+(' Validation | %0.4f | %0.2f | %2.2f'%(self.validation.loss, stat, self.validation.roc.auc*100))+' |'+('#'*bar)+'| '+overtrain
        self.logprint(s, end=' ')

        try:
            self.trainingHistory['validation.stat'].append(copy(stat))
            self.trainingHistory['validation.loss'].append(copy(self.validation.loss))
            self.trainingHistory['validation.auc'].append(copy(self.validation.roc.auc))
        except KeyError:
            self.trainingHistory['validation.stat'] = [copy(stat)]
            self.trainingHistory['validation.loss'] = [copy(self.validation.loss)]
            self.trainingHistory['validation.auc'] = [copy(self.validation.roc.auc)]

    def train(self):
        self.net.train()
        print_step = len(self.training.trainLoader)//200+1

        totalLoss = 0
        totalttError = 0
        totalLargeReweightLoss = 0
        rMax=0
        startTime = time.time()
        backpropTime = 0
        for i, (J, O, D, Q, y, w) in enumerate(self.training.trainLoader):
            J, O, D, Q, y, w = J.to(device), O.to(device), D.to(device), Q.to(device), y.to(device), w.to(device)
            self.optimizer.zero_grad()
            logits, quadjet_scores = self.net(J, O, D, Q)
            w_sum = w.sum()

            #compute classification loss
            loss  = (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).sum(dim=0)/w_sum#.mean(dim=0)

            if self.classifier in ["FvT", "DvT3"]:
                # compute loss term to account for failure to always give data higher prob than ttbar
                y_pred = F.softmax(logits, dim=-1)

                # compute loss term to penalize reweight factors greater than some threshold
                m4 = y_pred[:,d4.index]##(y_pred[:,d4.index] - y_pred[:,t4.index]).abs() # + eps
                r = m4/y_pred[:,d3.index] # m4/m3

                # only penlize three-tag events because fourtag events do not get reweighted
                r[y==d4.index] *= 0

                rMax = torch.max(r) if torch.max(r)>rMax else rMax

            #perform backprop
            backpropStart = time.time()
            loss.backward()
            self.optimizer.step()
            backpropTime += time.time() - backpropStart

            if not totalLoss: totalLoss = loss.item()
            totalLoss = totalLoss*0.98 + loss.item()*(1-0.98) # running average with 0.98 exponential decay rate
            if (i+1) % print_step == 0:
                elapsedTime = time.time() - startTime
                fractionDone = float(i+1)/len(self.training.trainLoader)
                percentDone = fractionDone*100
                estimatedEpochTime = elapsedTime/fractionDone
                timeRemaining = estimatedEpochTime * (1-fractionDone)
                estimatedBackpropTime = backpropTime/fractionDone
                progressString = str(('\rTraining %3.0f%% ('+loadCycler.next()+')  Loss: %0.4f | Time Remaining: %3.0fs | Estimated Epoch Time: %3.0fs | Estimated Backprop Time: %3.0fs ')%
                                     (percentDone, totalLoss, timeRemaining, estimatedEpochTime, estimatedBackpropTime))

                if self.classifier in ['FvT', 'DvT3']:
                    progressString += str(('| rMax %0.1f ')%(rMax))

                if quadjet_scores is not None:
                    q_1234, q_1324, q_1423 = quadjet_scores[-1,0], quadjet_scores[-1,1], quadjet_scores[-1,2]
                    progressString += str(('| q_score[-1] = (%0.2f, %0.2f, %0.2f)   ')%(q_1234,q_1324, q_1423))

                sys.stdout.write(progressString)
                sys.stdout.flush()

        self.trainEvaluate()

    def trainEvaluate(self, doROC=True):
        self.evaluate(self.training, doROC=doROC)
        sys.stdout.write(' '*150)
        sys.stdout.flush()
        bar=self.training.roc.auc
        bar=int((bar-self.barMin)*self.barScale) if bar > self.barMin else 0
        stat = self.training.norm_d4_over_B if self.classifier == 'FvT' else self.training.roc.maxSigma
        print('\r',end='')
        s=' '*len(self.epochString())+('   Training | %0.4f | %0.2f | %2.2f'%(self.training.loss, stat, self.training.roc.auc*100))+" |"+("-"*bar)+"|"
        self.logprint(s)

        try:
            self.trainingHistory['training.stat'].append(copy(stat))
            self.trainingHistory['training.loss'].append(copy(self.training.loss))
            self.trainingHistory['training.auc'].append(copy(self.training.roc.auc))
        except KeyError:
            self.trainingHistory['training.stat'] = [copy(stat)]
            self.trainingHistory['training.loss'] = [copy(self.training.loss)]
            self.trainingHistory['training.auc'] = [copy(self.training.roc.auc)]


    def saveModel(self, writeFile=True):
##        self.model_dict = {'model': deepcopy(model.net.state_dict()), 'optimizer': deepcopy(model.optimizer.state_dict()), 'epoch': self.epoch}
        self.model_dict = {'model': deepcopy(self.net.state_dict()),
                           'optimizer': deepcopy(self.optimizer.state_dict()),
                           'epoch': self.epoch,
                           'training history': copy(self.trainingHistory)}
        if writeFile:
            self.modelPkl = self.modelPath + '%s_epoch%d_loss%.4f.pkl'%(self.name, self.epoch, self.validation.loss)
            self.logprint('* '+self.modelPkl)
            torch.save(self.model_dict, self.modelPkl)

    def loadModel(self):
        self.net.load_state_dict(self.model_dict['model']) # load model from previous saved state
        self.optimizer.load_state_dict(self.model_dict['optimizer'])
        self.epoch = self.model_dict['epoch']
        self.logprint("Revert to epoch %d"%self.epoch)


    def makePlots(self):
        if self.classifier in ['SvB']:
            plotROC(self.training.roc,    self.validation.roc,    plotName=self.modelPkl.replace('.pkl', '_ROC.pdf'))

        if self.classifier in ['DvT3']:
            plotROC(self.training.roc_t3, self.validation.roc_t3, plotName=self.modelPkl.replace('.pkl', '_ROC_t3.pdf'))

        if self.classifier in ['FvT','DvT4']:
##            plotROC(self.training.roc_td, self.validation.roc_td, plotName=self.modelPkl.replace('.pkl', '_ROC_td.pdf'))
            plotROC(self.training.roc_43, self.validation.roc_43, plotName=self.modelPkl.replace('.pkl', '_ROC_43.pdf'))


        plotClasses(self.training, self.validation, self.modelPkl.replace('.pkl', '.pdf'), self.classes)

    def runEpoch(self, print_all_epochs=False):
        self.epoch += 1

        self.train()
        self.validate()

        self.train_losses.append(copy(self.training  .loss))
        self.valid_losses.append(copy(self.validation.loss))
        self.train_aucs.append(copy(self.training  .roc.auc))
        self.valid_aucs.append(copy(self.validation.roc.auc))
        if self.classifier in ['FvT']:
            self.train_stats.append(copy(self.training  .norm_d4_over_B))
            self.valid_stats.append(copy(self.validation.norm_d4_over_B))
        if self.classifier in ['SvB', 'SvB_MA']:
            self.train_stats.append(copy(self.training  .roc.maxSigma))
            self.valid_stats.append(copy(self.validation.roc.maxSigma))

        self.plotTrainingProgress()

        if print_all_epochs or (self.training.loss < self.training.loss_best or (abs(self.validation.norm_d4_over_B-1)<0.009 and abs(self.training.norm_d4_over_B-1)<0.009)):
            if self.training.loss < self.training.loss_best:
                self.foundNewBest = True
                self.training.loss_best = copy(self.training.loss)

            self.saveModel()
            self.makePlots()
            ## This is where you would save predicted probabilities. np.save("x.npy", self.validation.y_pred)
        else:
            self.logprint('')

        if self.fixedSchedule:
            self.scheduler.step()
            if self.epoch in self.bs_milestones:
                self.incrementTrainLoader()
            if self.epoch in self.lr_milestones:
                self.lr_change.append(self.epoch+0.5)
                self.logprint("Decay learning rate")
        elif not self.training.trainLoaders: # ran out of increasing batch size, start dropping learning rate instead
            self.scheduler.step(self.training.loss)        
        elif self.training.loss > self.training.loss_min and self.training.trainLoaders:
            if self.patience == self.max_patience:
                self.patience = 0
                self.incrementTrainLoader()
            else:
                self.patience += 1
        else:
            self.patience = 0

            
    def runEpochs(self, print_all_epochs=False):
        for i in range(self.epochs):
            self.runEpoch(print_all_epochs=print_all_epochs)
    

    def incrementTrainLoader(self):
        try:
            batchString = 'Increase training batch size: %i -> %i (%i batches)'%(self.training.trainLoader.batch_size, self.training.trainLoaders[-1].batch_size, len(self.training.trainLoaders[-1]) )
            self.logprint(batchString)
            self.training.trainLoader = self.training.trainLoaders.pop()
            self.bs_change.append(self.epoch+0.5)
        except IndexError:
            batchString = 'Ran out of training data loaders'
            self.logprint(batchString)
            
    def dump(self):
        print(self.net)
        self.net.layers.print()
        print(self.name)
        print('pDropout:',self.pDropout)
        print('lrInit:',self.lrInit)
        print('startingEpoch:',self.startingEpoch)
        print('loss_best:',self.training.loss_best)
        self.nTrainableParameters = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('N trainable params:',self.nTrainableParameters)



    def predict(self, df):
##        df = pd.read_hdf(fileName, key='df')
##        yearIndex = fileName.find('201')
##        year = float(fileName[yearIndex:yearIndex+4])
##        df['year'] = pd.Series(year*np.ones(df.shape[0], dtype=np.float32), index=df.index)
##
        df['pseudoTagWeight'] = 1
        df['mcPseudoTagWeight'] = df['weight']
        ##df['xW'] = 1
        ##df['xbW']=1
        ##df['nSelJets'] = 1
        ##df['year'] = 2017
        ##df['xWt'] = 1
        ##df['nSelJets'] = 1

        """
        print("____ DF SHAPE ______")
        print(df.shape[0])
        
        df.drop(df[df['weight'] == 0].index, inplace=True)
    
        print("EFFN ", n)
        """
        n = df.shape[0]
  
        J, O, D, Q, y, w = self.dfToTensors(df, y_true=self.yTrueLabel)

        # Set up data loaders
        print("Make data loader")
        dset   = TensorDataset(J, D, D, Q, y, w)
        #updateResults = loaderResults("update")
        #updateResults.evalLoader = 
        ldr  = DataLoader(dataset=dset, batch_size=eval_batch_size, shuffle=False, num_workers=n_queue, pin_memory=True)
        #updateResults.n = n

        self.net.eval()
        y_pred, y_true, w_ordered = np.ndarray((n,2), dtype=np.float), np.zeros(n, dtype=np.float), np.zeros(n, dtype=np.float)
        q_score = np.ndarray((n, 3), dtype=np.float)
        print_step = len(ldr)//200+1
        nProcessed = 0
        loss = 0
        for i, (J, O, D, Q, y, w) in enumerate(ldr):##enumerate(results.evalLoader):
            nBatch = w.shape[0]
            J, O, D, Q, y, w = J.to(device), O.to(device), D.to(device), Q.to(device), y.to(device), w.to(device)
            logits, quadjet_scores = self.net(J, O, D, Q)
            loss += (w * F.cross_entropy(logits, y, weight=wC, reduction='none')).sum(dim=0).cpu().item()
            y_pred[nProcessed:nProcessed+nBatch] = F.softmax(logits, dim=-1).detach().cpu().numpy()
            y_true[nProcessed:nProcessed+nBatch] = y.cpu()
            q_score[nProcessed:nProcessed+nBatch] = quadjet_scores.detach().cpu().numpy()
            w_ordered[nProcessed:nProcessed+nBatch] = w.cpu()

            nProcessed+=nBatch
            if int(i+1) % print_step == 0:
                percent = float(i+1)*100/len(ldr)
                sys.stdout.write('\rEvaluating %3.0f%%     '%(percent))
                sys.stdout.flush()

        return y_pred, y_true

    def plotByEpoch(self, train, valid, ylabel, suffix, loc='best'):
        fig = plt.figure(figsize=(10,7))

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        #plt.ylim(yMin,yMax)
        x = np.arange(1,self.epoch+1)
        plt.plot(x, train,
                 marker="o",
                 linestyle="-",
                 linewidth=1, alpha=1.0,
                 color="#d34031",
                 label="Training")
        plt.plot(x, valid,
                 marker="o",
                 linestyle="-",
                 linewidth=2, alpha=0.5,
                 color="#d34031",
                 label="Validation")
        plt.xticks(x)
        #plt.yticks(np.linspace(-1, 1, 5))
        plt.legend(loc=loc)

        xlim = plt.gca().get_xlim()
        ylim = plt.gca().get_ylim()

        for e in self.bs_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        for e in self.lr_change:
            plt.plot([e,e], ylim, color='k', alpha=0.5, linestyle='--', linewidth=1)
        if 'norm' in suffix:
            plt.plot(xlim, [1,1], color='k', alpha=0.5, linestyle='-', linewidth=1)
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)

        plotName = self.modelPath+'%s_%s.pdf'%(self.name, suffix)
        try:
            fig.savefig(plotName)
        except:
            print("Cannot save fig: ",plotName)
        plt.close(fig)

    def plotTrainingProgress(self):
        self.plotByEpoch(self.train_losses, self.valid_losses, "Loss", 'loss', loc='upper right')
        self.plotByEpoch(self.train_aucs,   self.valid_aucs,   "AUC",  'auc',  loc='lower right')
        if self.classifier in ['FvT']:
            self.plotByEpoch(self.train_stats,  self.valid_stats, "Data / Background",    'norm',  loc='best')
        if self.classifier in ['SvB', 'SvB_MA']:
            self.plotByEpoch(self.train_stats,  self.valid_stats, "Sensitivity Estimate", 'sigma', loc='lower right')


#Simple ROC Curve plot function
def plotROC(train, val, plotName): #fpr = false positive rate, tpr = true positive rate
    f = plt.figure()
    ax = plt.subplot(1,1,1)
    plt.subplots_adjust(left=0.1, top=0.95, right=0.95)

    #y=-x diagonal reference curve for zero mutual information ROC
    ax.plot([0,1], [1,0], color='k', alpha=0.5, linestyle='--', linewidth=1)

    plt.xlabel('Rate( '+val.trueName+' to '+val.trueName+' )')
    plt.ylabel('Rate( '+val.falseName+' to '+val.falseName+' )')
    bbox = dict(boxstyle='square',facecolor='w', alpha=0.8, linewidth=0.5)
    ax.plot(train.tpr, 1-train.fpr, color='#d34031', linestyle='-', linewidth=1, alpha=1.0, label="Training")
    ax.plot(val  .tpr, 1-val  .fpr, color='#d34031', linestyle='-', linewidth=2, alpha=0.5, label="Validation")
    ax.legend(loc='lower left')

    if hasattr(val, 'auc_ci'):
        print(val.auc_ci)
#        ax.text(0.56, 1.07, "Validation AUC: %0.4f, [%0.4f, %0.4f]"%(val.auc, val.auc_ci[0], val.auc_ci[1]))

        ax.text(0.65, 1.07, "Validation AUC: %0.4f, [%0.4f, %0.4f]"%(val.auc, val.auc_ci[0], val.auc_ci[1]))

    else:
        ax.text(0.73, 1.07, "Validation AUC = %0.4f"%(val.auc))

##    if val.maxSigma is not None:
##        #ax.scatter(rate_StoS, rate_BtoB, marker='o', c='k')
##        #ax.text(rate_StoS+0.03, rate_BtoB-0.100, ZB+"SR \n (%0.2f, %0.2f)"%(rate_StoS, rate_BtoB), bbox=bbox)
##        ##ax.scatter(val.tprMaxSigma, (1-val.fprMaxSigma), marker='o', c='#d34031')
##        ##ax.text(val.tprMaxSigma+0.03, (1-val.fprMaxSigma)-0.025, 
##        ##        ("(%0.3f, %0.3f), "+val.pName+" $>$ %0.2f \n S=%0.1f, B=%0.1f, $%1.2f\sigma$")%(val.tprMaxSigma, (1-val.fprMaxSigma), val.thrMaxSigma, val.S, val.B, val.maxSigma), 
##                bbox=bbox)

    plt.figure(figsize=(20,10))
    f.savefig(plotName)
    plt.close(f)

def plotClasses(train, valid, name, classes):
    # Make place holder datasets to add the training/validation set graphical distinction to the legend
    trainLegend=pltHelper.dataSet(name=  'Training Set', color='black', alpha=1.0, linewidth=1)
    validLegend=pltHelper.dataSet(name='Validation Set', color='black', alpha=0.5, linewidth=2)

    extraClasses = []
    if classes[0].abbreviation == "sg":
        ##extraClasses = [sg,bg]
        binMin, binMax =  0, 21
        bins = [b/(binMax-binMin) for b in range(binMin,binMax)]
    else:
        binMin, binMax = -5, 21
        bins = [b/(binMax-binMin) for b in range(binMin,binMax)]

    for cl1 in classes: # loop over classes    ######## This for loop used to fail
        cl1cl2_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': 'P( '+cl1.name+r' $\rightarrow$ Class )',
                       'ylabel': 'Arb. Units',
                       }
        cl2cl1_args = {'dataSets': [trainLegend,validLegend],
                       'bins': bins,
                       'xlabel': r'P( Class $\rightarrow$ '+cl1.name+' )',
                       'ylabel': 'Arb. Units',
                       }
        for cl2 in classes: # loop over classes
        # Make datasets to be plotted
            cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl1cl2_args['dataSets'] += [cl1cl2_valid, cl1cl2_train]

            cl2cl1_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(train,'w'+cl2.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
            cl2cl1_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl2.abbreviation+cl1.abbreviation), weights=getattr(valid,'w'+cl2.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
            cl2cl1_args['dataSets'] += [cl2cl1_valid, cl2cl1_train]

        if classes[0].abbreviation != "sg":
            # multijet probabilities well defined but no multijet class labels. Therefore cl1cl2 plot can include multijet but not cl2cl1 plot.
            m4 = classInfo(abbreviation='m4', name= 'FourTag Multijet', color='blue')
            m3 = classInfo(abbreviation='m3', name='ThreeTag Multijet', color='violet')
            for cl2 in [m4,m3]:
                cl1cl2_train = pltHelper.dataSet(name=cl2.name, points=getattr(train,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(train,'w'+cl1.abbreviation)/train.w_sum, color=cl2.color, alpha=1.0, linewidth=1)
                cl1cl2_valid = pltHelper.dataSet(               points=getattr(valid,'p'+cl1.abbreviation+cl2.abbreviation), weights=getattr(valid,'w'+cl1.abbreviation)/valid.w_sum, color=cl2.color, alpha=0.5, linewidth=2)
                cl1cl2_args['dataSets'] += [cl1cl2_train, cl1cl2_valid]

        #make the plotter
        cl1cl2 = pltHelper.histPlotter(**cl1cl2_args)
        cl2cl1 = pltHelper.histPlotter(**cl2cl1_args)

        #remove the lines from the trainLegend/validLegend placeholders
        cl1cl2.artists[0].remove()
        cl1cl2.artists[1].remove()
        cl2cl1.artists[0].remove()
        cl2cl1.artists[1].remove()

        #save the pdf
        cl1cl2.savefig(name.replace('.pdf','_'+cl1.abbreviation+'_to_class.pdf'))
        cl2cl1.savefig(name.replace('.pdf','_class_to_'+cl1.abbreviation+'.pdf'))

    if classes[0].abbreviation != "sg":
        bm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                         'bins': [b/20.0 for b in range(-10,61)],
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        d4_train = pltHelper.dataSet(name=d4.name, points=train.rd4, weights=   train.wd4/train.w_sum, color=d4.color, alpha=1.0, linewidth=1)
        d4_valid = pltHelper.dataSet(              points=valid.rd4, weights=   valid.wd4/valid.w_sum, color=d4.color, alpha=0.5, linewidth=2)
        bm_train = pltHelper.dataSet(name='Background Model', 
                                     points=train.rd3, 
                                     weights=train.wd3, 
                                     ##points=np.concatenate((train.rd3,train.rt3,train.rt4),axis=None), 
                                     ##weights=np.concatenate((train.wd3,-train.wt3,train.wt4)/train.w_sum,axis=None), 
                                     color='brown', alpha=1.0, linewidth=1)
        bm_valid = pltHelper.dataSet(points=valid.rd3, 
                                     weights=valid.wd3/train.w_sum, 
                                     color='brown', alpha=0.5, linewidth=2)
##        t4_train = pltHelper.dataSet(name=t4.name, points=train.rt4, weights= train.wt4/train.w_sum, color=t4.color, alpha=1.0, linewidth=1)
##        t4_valid = pltHelper.dataSet(              points=valid.rt4, weights= valid.wt4/valid.w_sum, color=t4.color, alpha=0.5, linewidth=2)
##        t3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
##        t3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
        bm_vs_d4_args['dataSets'] += [d4_valid, d4_train, bm_valid, bm_train]##, t4_valid, t4_train, t3_valid, t3_train]
        bm_vs_d4 = pltHelper.histPlotter(**bm_vs_d4_args)
        bm_vs_d4.artists[0].remove()
        bm_vs_d4.artists[1].remove()
        bm_vs_d4.savefig(name.replace('.pdf','_bm_vs_d4.pdf'))


        rbm_vs_d4_args = {'dataSets': [trainLegend,validLegend],
                         'bins': [b/20.0 for b in range(-10,61)],
                         'xlabel': r'P( Class $\rightarrow$ FourTag Multijet )/P( Class $\rightarrow$ ThreeTag Data )',
                         'ylabel': 'Arb. Units',
                         }
        rbm_train = pltHelper.dataSet(name='Background Model', 
                                     points=train.rd3, 
                                     weights=train.rd3*train.wd3/train.w_sum, 
                                     ##points=np.concatenate((train.rd3,train.rt4),axis=None), 
                                     ##weights=np.concatenate((train.rd3*train.wd3,train.wt4)/train.w_sum,axis=None), 
                                     color='brown', alpha=1.0, linewidth=1)
        rbm_valid = pltHelper.dataSet(points=valid.rd3, 
                                     weights=valid.rd3*valid.wd3/valid.w_sum, 
                                     color='brown', alpha=0.5, linewidth=2)
##        rt3_train = pltHelper.dataSet(name=t3.name, points=train.rt3, weights=-train.rt3*train.wt3/train.w_sum, color=t3.color, alpha=1.0, linewidth=1)
##        rt3_valid = pltHelper.dataSet(              points=valid.rt3, weights=-valid.rt3*valid.wt3/valid.w_sum, color=t3.color, alpha=0.5, linewidth=2)
        rbm_vs_d4_args['dataSets'] += [d4_valid, d4_train, rbm_valid, rbm_train]##, t4_valid, t4_train, rt3_valid, rt3_train]
        rbm_vs_d4 = pltHelper.histPlotter(**rbm_vs_d4_args)
        rbm_vs_d4.artists[0].remove()
        rbm_vs_d4.artists[1].remove()
        rbm_vs_d4.savefig(name.replace('.pdf','_rbm_vs_d4.pdf'))


"""
df3b = pd.read_hdf("bbbj.h5")[0:6000]
df4b = pd.read_hdf("bbbb.h5")[0:6000]
##df_main = pd.read_hdf("bbbj.h5")[0:10000]
##df_main2 = pd.read_hdf("bbbb.h5")[0:10000]
##df_main = pd.concat([df_main, df_main2])
model = modelParameters(dfB=df3b, dfS=df4b, fileName="")
#model initial state
print("Setup training/validation tensors")
model.trainSetup()
model.makePlots()
# Training loop
#if classifier in ['FvT','DvT4']:
#    model.net.layers.setLayerRequiresGrad(range(1,5), False)
#    model.net.layers.initLayer(range(14,15))
for e in range(epochs): 
    # if classifier in ['DvT4'] and e == 5:
    #     model.net.layers.setLayerRequiresGrad(range(6,9), True)
    model.runEpoch()
print()
print(">> DONE <<")
if model.foundNewBest: print("Minimum Loss =", model.training.loss_best)
"""
