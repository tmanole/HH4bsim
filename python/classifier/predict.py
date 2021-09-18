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

from training import modelParameters, loaderResults

import argparse

yTrueLabel = 'fourTag'

print_step=10


df1 = pd.read_hdf('../../dataframes/full.h5', 'df')

df_pred = df1[(df1['SR']==True) & (df1['fourTag']==False)]

model = modelParameters('pytorchModels/FvT_ResNet_6_10_11_lr0.001_epochs5_stdscale_epoch1_auc1.0000.pkl')

model.dfToTensors(df_pred, y_true = yTrueLabel)

X_pred,   P_pred  , O_pred  , D_pred  , Q_pred  , A_pred  , y_pred  , w_pred   = model.dfToTensors(df_pred, y_true=yTrueLabel)
X_pred   = torch.FloatTensor(model.scalers['xVariables'].transform(X_pred))
for jet in range(P_pred.shape[2]):
    P_pred  [:,:,jet] = torch.FloatTensor(model.scalers[0].transform(P_pred  [:,:,jet]))
D_pred   = torch.FloatTensor(model.scalers['dijetAncillary'].transform(D_pred))
Q_pred   = torch.FloatTensor(model.scalers['quadjetAncillary'].transform(Q_pred))
A_pred   = torch.FloatTensor(model.scalers['ancillary'].transform(A_pred))

# Set up data loaders
dset_pred     = TensorDataset(X_pred,   P_pred,   P_pred,   D_pred,   Q_pred,   A_pred,   y_pred,   w_pred)


ldr     = DataLoader(dataset=dset_pred, batch_size=20)

device=torch.device('cpu')
def evaluate(model, ldr, doROC=True):
        model.net.eval()
        y_pred, y_true, w_ordered = [], [], []
        for i, (X, P, O, D, Q, A, y, w) in enumerate(ldr):
            X, P, O, D, Q, A, y, w = X.to(device), P.to(device), O.to(device), D.to(device), Q.to(device), A.to(device), y.to(device), w.to(device)
            logits = model.net(X, P, O, D, Q, A)
            prob_pred = torch.sigmoid(logits)
            y_pred.append(prob_pred.tolist())
            y_true.append(y.tolist())
            w_ordered.append(w.tolist())
            if (i+1) % print_step == 0:
                sys.stdout.write('\rEvaluating %3.0f%%     '%(float(i+1)*100/len(ldr)))
                sys.stdout.flush()

        y_prob = np.transpose(np.concatenate(y_pred))[0]
        y_pred = [1 if y_prob[i] >= 0.5 else 0 for i in range(len(y_prob))]
        y_true = np.transpose(np.concatenate(y_true))[0]
        #ldr.w      = np.transpose(np.concatenate(w_ordered))[0]
#        if doROC:
#            results.fpr, results.tpr, results.thr = roc_curve(results.y_true, results.y_pred, sample_weight=results.w)
#            results.roc_auc_prev = copy(results.roc_auc)
#            results.roc_auc = auc(results.fpr, results.tpr)
#            if results.roc_auc_prev:
#                if results.roc_auc_prev > results.roc_auc: results.roc_auc_decreased += 1
        return y_prob, y_pred, y_true


y_prob, y_pred, y_true = evaluate(model, ldr, False)

out = sum(y_pred == y_true)/len(y_true)
print("!")
print(out)
print("!")

probs = np.array(y_prob)

np.save("predictions/pred1.npy", probs)



