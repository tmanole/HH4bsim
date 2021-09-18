import numpy as np
import scipy.stats

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

