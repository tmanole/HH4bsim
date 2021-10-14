import numpy as np
import matplotlib.pyplot as plt
import emdFullCalc
from energyflow.emd import emd

import imp
try:
    imp.find_module("ROOT")
    import ROOT

except ImportError:
    import sys
    sys.path.insert(0, "../../../../ROOT/build/lib")
    import ROOT_copy as ROOT



#
# Following is From
# https://rivet.hepforge.org/code/1.8.2/a00677_source.html
#
# Do the general case thrust calculation
def calcT(listOfTV2s):
    # This function implements the iterative algorithm as described in the
    # Pythia manual. We take eight (four) different starting vectors
    # constructed from the four (three) leading particles to make sure that
    # we don't find a local maximum.
    p = listOfTV2s
    assert(len(p) >= 3)
    n = 4

    tvec = []
    tval = []
    
    nMin = 8

    for i in range(nMin):
        # Create an initial vector from the leading four jets
        foo = ROOT.TVector2(0,0)
        sign = i
        for k in range(n):
            if (sign % 2) == 1 :
                foo += p[k]
            else :
                foo -= p[k]
            sign = sign/2
                
        foo=foo.Unit()
        
        # Iterate
        diff=999.
        while (diff>1e-5):
            foobar = ROOT.TVector2(0,0)
            for k in range(len(p)):
                if (foo *p[k])>0 :
                    foobar+=p[k]
                else:
                    foobar-=p[k]
            diff=(foo-foobar.Unit()).Mod()
            foo=foobar.Unit()
        # while
            
        # Calculate the thrust value for the vector we found
        t=0.
        for k in range(len(p)):
            t+=abs(foo*p[k])
            
        #Store everything
        tval.append(t)
        tvec.append(foo)
    # for i in range(8)
        
        
    # Pick the solution with the largest thrust
    t=0.
    taxis = None
    for i in range(len(tvec)):
        if (tval[i]>t):
            t=tval[i]
            taxis=tvec[i]
    return t, taxis

    
#
# Do the full calculation
#

# returns TVector2
def calcThrust(listOfTV2s, debug = False, doSign=False):
    if(debug): print("Number of particles = ", jetPts.size())

    assert len(listOfTV2s) > 3
    
    # Get thrust
    val, axis = calcT(listOfTV2s)

    axis = axis.Unit();

    if doSign:
        val_Ta = 0
        val_TaSign = 0
        valSign = 0
        for i in listOfTV2s:
            val_Ta += abs(i.Mod()*np.sin(axis.Phi()-i.Phi()))
            val_TaSign += i.Mod()*np.sin(axis.Phi()-i.Phi())
            valSign += i*axis
        print(val_Ta/val,valSign/val,val_TaSign/val,val_TaSign/val_Ta,"(",val,",",val_Ta,")")

    if debug : print( "Axis = ",axis.X(), " ", axis.Y())
    return axis



# Returns TVector2
def getThrustAxis(jetList,doSign=False):
    phis = jetList[:,2]
    pxs = jetList[:,0]*np.cos(phis)
    pys = jetList[:,0]*np.sin(phis)

    jetPtsVecs = []
    for i in range(len(pxs)):
        jetPtsVecs.append(ROOT.TVector2(pxs[i],pys[i]))

        
    return calcThrust(jetPtsVecs,doSign=doSign);


def testThrust(ev0):  #ev0  = np.reshape(CR2b[1,:], (4,3))
    
    phi0 = getThrustAxis(ev0).Phi() % (2*np.pi)

    phiScan = np.linspace(0,2*np.pi,100)
    thrustVals = []
    for p in phiScan:
        thisTrustVal = 0
        for jet in ev0:
            thisTrustVal += jet[0]*abs(np.cos(jet[2]-p))
        thrustVals.append(thisTrustVal)
        

    plt.plot(phiScan,thrustVals,"k-")
    plt.xlabel("$\phi$")
    plt.ylabel("$\sum P_T$ along $\phi$")
    plt.plot([phi0,phi0],[np.min(thrustVals),np.max(thrustVals)],"r:",label="Thrust Axis")
    plt.legend(loc="best")
    
    plt.show()


    



def compThrustToTrueMin(ev0,ev1,R=2*np.pi,periodic_phi=True):
    phiRange = np.linspace(0,2*np.pi,100)
    emdVsPhi = []

    tAxis0 = getThrustAxis(ev0)
    tAxis1 = getThrustAxis(ev1)
    #dPhiThrust = tAxis0.DeltaPhi(tAxis1) % (2*np.pi)
    #dPhiThrust_pi = (tAxis0.DeltaPhi(tAxis1) + np.pi) % (2*np.pi)
    dPhiThrust = (-1*tAxis0.DeltaPhi(tAxis1)) % (2*np.pi)
    dPhiThrust_pi = (dPhiThrust+np.pi) % (2*np.pi)

    for dPhi in phiRange:

        thisEmd = emd(emdFullCalc.movePhi(ev0,dPhi),ev1,R,periodic_phi=periodic_phi)
        emdVsPhi.append(thisEmd)



    plt.plot(phiRange,emdVsPhi)
    plt.plot([dPhiThrust,dPhiThrust],[np.min(emdVsPhi),np.max(emdVsPhi)],color="r",label="thrust $\Delta \phi$")
    plt.plot([dPhiThrust_pi,dPhiThrust_pi],[np.min(emdVsPhi),np.max(emdVsPhi)],"r:",label="thrust $\Delta \phi + \pi$")
    #plt.plot([thrustPhi,thrustPhi],[np.min(emdVsPhi),np.max(emdVsPhi)],color="r",label="thrust $\Delta \phi$")
    #plt.plot([thrustPhi2,thrustPhi2],[np.min(emdVsPhi),np.max(emdVsPhi)],"r:",label="thrust $\Delta \phi + \pi$")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0.)


    plt.show()
