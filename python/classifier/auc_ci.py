import numpy as np
import scipy.stats
from roc_auc_with_negative_weights import roc_auc_with_negative_weights
import matplotlib.pyplot as plt


def bootstrap(x, y, weights, alpha=0.05, B=1000, o=0):
    """ Compute a bootstrap confidence interval for the weighted AUC.

        Args:
            x(Numpy array): Classification probabilities for the class 0 (typically "3b" or "CR").
            y(Numpy array): Classification probabilities for the class 1 (typically "4b" or "SR").
            w(Numpy array): Weights.
            auc(float): Estimate of the area under ROC curve.
            alpha(float): Confidence interval level.

        Returns:
            List: Lower and upper confidence bounds for the AUC.
    """
    n = x.shape[0]

    auc = roc_auc_with_negative_weights(x, y, weights=weights)
    print(auc) 
    aucs = []

    tinds = np.empty((B,n))

    for b in range(B):
        print("boot ",  b)
        inds = np.random.choice(a=n, size=n, replace=True).flatten().tolist()#, p=weights/np.sum(weights)).flatten().tolist()
#        print(inds) 
#        print(x.shape)
#        print(inds)


        """
        for i in range(n):
            if weights[inds[i]] == 0:
                print("offensive ind: ", i)
        """
        print("Unique inds: ", np.unique(inds).shape)
        tinds[b,inds] += 1

        xx = x[inds].flatten()
        yy = y[inds].flatten()
        ww = weights[inds].flatten()
#        print(xx.shape)
#        print(np.sum(x[weights > 0])) 
#        print(np.sum(xx[ww > 0])) 

        aucs.append(roc_auc_with_negative_weights(xx, yy, weights=ww))
            
#    print("uniques")
#    print(np.unique(tinds, axis=0))
#    print(np.unique(tinds, axis=1))
#
#    print(np.sum(tinds[b,:]==0))
#    print(np.sum(np.unique(tinds, axis=1)==0))
#
#    print("@")
#    print(np.where(~tinds.transpose().any(axis=1))[0].shape)
#    print(np.sum(weights==0))

#    print("offensive inds")
#    for i in range(n):g
#        if tinds[0,i] > 0 and weights[i] == 0:
#            print(i)
#

    plt.hist(aucs)
    plt.savefig("boot" + str(o) + ".pdf") 

    se = np.sqrt(np.var(aucs))
    print("Standard error: ", se)

    q = scipy.stats.norm.ppf(1-alpha/2)

    return [auc - q * se, auc + q * se]


x1 = np.load("../classes1.npy")
y1 = np.load("../preds1.npy")
w1 = np.load("../weights1.npy")

x2 = np.load("../classes10.npy")
y2 = np.load("../preds10.npy")
w2 = np.load("../weights10.npy")

x3 = np.load("../classes100.npy")
y3 = np.load("../preds100.npy")
w3 = np.load("../weights100.npy")

print(np.sum(w1<1e-9))
print(np.sum(w2<1e-9))
print(np.sum(w3<1e-9))
print(w1.shape)
print(w2.shape)
print(w3.shape)
ci1 = bootstrap(x1, y1, w1, o=1)
ci2 = bootstrap(x2, y2, w2, o=2)
ci3 = bootstrap(x3, y3, w3, o=3)

print(ci1)
print(ci2)
print(ci3)

print(ci1[1] - ci1[0])
print(ci2[1] - ci2[0])
print(ci3[1] - ci3[0])


def bamber(x, y, auc, alpha=0.05):
    """ Compute the Bamber(1975) Wald CI for the AUC.

        Args:
            x(Numpy array): Classification probabilities for the class 0 (typically "3b" or "CR").
            y(Numpy array): Classification probabilities for the class 1 (typically "4b" or "SR").
            auc(float): Estimate of the area under ROC curve.
            alpha(float): Confidence interval level.

        Returns:
            List: Lower and upper confidence bounds for the AUC.
    """

    print("Bamber starting.")


    nx = x.size
    ny = y.size

#    p = 0
#    for xi in np.nditer(x):
#        for yj in np.nditer(y):
#            p += 1 if xi != yj else 0
#
#    p /= nx * ny

    p = 1

#    v_j = np.zeros(nx)
#    for i in range(nx):  
#        for yi in np.nditer(y):
#            if yi == x[i]:
#                v_j[i] += 0.5
#
#            if yi > x[i]:
#                v_j[i] += 1

#    for i in range(nx):
    v_j = np.array([0.5 * np.sum(y == x[i]) + np.sum(y > x[i]) for i in range(nx)])

    u_j = ny - v_j

    vj_ = np.zeros(ny)
#    for i in range(ny):
#        for xi in np.nditer(x):
#            if y[i] == xi:
#                vj_[i] += 0.5
#
#            if y[i] > xi:
#                vj_[i] += 1

#    for j in range(ny):
    vj_ = np.array([0.5 * np.sum(x == y[j])  + np.sum(x < y[j]) for j in range(ny)])

    uj_ = nx - vj_

#    bXXY = 0
#    for i in range(ny):
#        bXXY += uj_[i] * (uj_[i] - 1) + vj_[i] * (vj_[i] - 1) - 2 * uj_[i] * vj_[i]
    bXXY = np.sum(uj_ * (uj_ - 1) + vj_ * (vj_ - 1) - 2 * uj_ * vj_)

    bXXY /= nx * (nx-1) * ny

#    bYYX = 0
#    for i in range(nx):
#        bYYX += u_j[i] * (u_j[i] - 1) + v_j[i] * (v_j[i] - 1) - 2 * u_j[i] * v_j[i]
    bYYX = np.sum(u_j * (u_j - 1) + v_j * (v_j - 1) - 2 * u_j * v_j)

    bYYX /= ny * (ny-1) * nx

    V = p + (nx-1) * bXXY + (ny-1) * bYYX - 4 * (nx + ny - 1) * (auc - 0.5)**2

    V /= 4 * (nx-1) * (ny-1)
    
    q = scipy.stats.norm.ppf(1-alpha/2)

#    print(V)
#    print(q)

    return [auc - q * np.sqrt(V), auc + q * np.sqrt(V)]

def hanley_mcneil1(x, y, auc, alpha=0.05):
    """ Compute the Hanley and McNeil Wald CI for the AUC.

        Args:
            x(Numpy array): Classification probabilities for the class 0 (typically "3b" or "CR").
            y(Numpy array): Classification probabilities for the class 1 (typically "4b" or "SR").
            auc(float): Estimate of the area under ROC curve.
            alpha(float): Confidence interval level.

        Returns:
            List: Lower and upper confidence bounds for the AUC.
    """
    nx = x.size
    ny = y.size

    p = 0
    for xi in np.nditer(x):
        for yj in np.nditer(y):
            p += 1 if xi == yj else 0

    p /= nx * ny

    Q1 = 0
    for xi in np.nditer(x):
        inner = 0

        for yj in np.nditer(y):
            if yj == xi:
                Q1 += 0.5
  
            if yj > xi:
                Q1 += 1

        Q1 += inner**2

    Q1 /= ny * (nx**2)

    Q2 = 0
    for yj in np.nditer(y):
        inner = 0

        for xi in np.nditer(x):
            if yj == xi:
                Q2 += 0.5
  
            if yj > xi:
                Q2 += 1

        Q2 += inner**2

    Q2 /= nx * ny**2


    V = auc * (1-auc) - 0.25 * p + (ny - 1) * (Q1 - auc**2) + (nx - 1) * (Q2 - auc**2)

    V /= (nx-1) * (ny - 1)
   
    q = scipy.stats.norm.ppf(1-alpha/2)

    print(V)
    print(q)
    print(Q1)
    print(Q2)


    return [auc - q * np.sqrt(V), auc + q * np.sqrt(V)]


#n = 1000
#
#x = np.random.uniform(0,0.7, size=n)
#y = np.random.uniform(size=n)
#
#auc = 0
#
#for i in range(n):
#    for j in range(n):
#        if x[i] < y[j]:
#            auc += 1
#
#auc /= n * n
#
#print(auc)
#
##hanley_mcneil1(x, y, auc)
#print(bamber(x, y, auc))

