import numpy as np
from energyflow.emd import emd
import matplotlib.pyplot as plt

import thrust 

def movePhi(inEvent,phi):
    outEvent = np.copy(inEvent)
    outEvent[:,2] += phi
    outEvent[:,2] = outEvent[:,2] % (2*np.pi)
    return outEvent

def flipEta(inEvent):
    outEvent = np.copy(inEvent)
    outEvent[:,1] *= -1
    return outEvent

def flipPhi(inEvent):
    outEvent = np.copy(inEvent)
    outEvent[:,2] *= -1
    return outEvent

def minEmdEtaFlip(ev0, ev1, R,periodic_phi=True):
    nom = emd(ev0,ev1,R,periodic_phi=periodic_phi)

    flipEtaEvent = flipEta(ev0)
    etaFlip = emd(flipEtaEvent,ev1,R,periodic_phi=periodic_phi)
    if nom < etaFlip:
        return nom, ev0
    return etaFlip, flipEtaEvent


def minEmdPhiRot(ev0,ev1,R,nPhiSteps=15,periodic_phi=True):
    phiRange = np.linspace(0,2*np.pi,nPhiSteps)
    emdVsPhi = []

    minEmd = 9999
    mindPhi = None
    for dPhi in phiRange:
        
        thisEmd = emd(movePhi(ev0,dPhi),ev1,R,periodic_phi=periodic_phi)
        emdVsPhi.append(thisEmd)
        if thisEmd < minEmd:
            minEmd = thisEmd
            mindPhi = dPhi

    return minEmd, movePhi(ev0,dPhi), mindPhi



def emdFull(ev0, ev1, R, nPhiSteps=15, periodic_phi=True, return_flow=False):

    #emdvalEta, eventMinEta = minEmdEtaFlip(ev0,ev1,R)

    results = []
    # Nominal min over phi
    results.append(minEmdPhiRot(ev0,ev1,R,nPhiSteps=nPhiSteps))

    # Flip Eta min over phi
    results.append(minEmdPhiRot(flipEta(ev0),ev1,R,nPhiSteps=nPhiSteps))

    # Flip Phi min over phi
    results.append(minEmdPhiRot(flipPhi(ev0),ev1,R,nPhiSteps=nPhiSteps))

    # Flip Phi and Eet min over phi
    results.append(minEmdPhiRot(flipEta(flipPhi(ev0)),ev1,R,nPhiSteps=nPhiSteps))

    
    emdval   = 9999
    eventMin = None
    phiMin   = None
    minItr   = None
    for relItr, res in enumerate(results):
        if res[0] < emdval:
            emdval = res[0]
            eventMin = res[1]
            phiMin = res[2]
            minItr = relItr

    if return_flow:
        emdval, emdFlow = emd(eventMin,ev1,R,return_flow=True,periodic_phi=periodic_phi)
        return emdval,emdFlow,eventMin, phiMin
    return emdval, eventMin, phiMin, relItr, results

#
# 
#
def emdFullAprox(ev0, ev1, R, sumPz0=None, sumPz1=None, tAxis1=None, tAxis0=None, tAxis0_fEta=None, tAxis0_fPhi=None, tAxis0_fPhifEta=None, periodic_phi=True, return_flow=False):

    
    if sumPz0 is None:
        sumPz0 = np.sum(ev0[:,0] * np.sinh(ev0[:,1]))
    if sumPz1 is None:
        sumPz1 = np.sum(ev1[:,0] * np.sinh(ev1[:,1]))
    if tAxis1 is None:
        tAxis1 = thrust.getThrustAxis(ev1)        

    #
    # Get Correct eta orientation
    #
    if sumPz0*sumPz1 < 0:  doEtaFlip = True
    else:                  doEtaFlip = False


    thrustCalcs = []

    if not doEtaFlip:
        if tAxis0 is None: tAxis0 = thrust.getThrustAxis(ev0)
        dPhiThrust = tAxis0.DeltaPhi(tAxis1) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(ev0,-dPhiThrust),ev1,R,periodic_phi=periodic_phi))
        dPhiThrust_pi = (tAxis0.DeltaPhi(tAxis1) + np.pi) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(ev0,-dPhiThrust_pi),ev1,R,periodic_phi=periodic_phi))
        
        evt0_fPhi   = flipPhi(ev0)
        if tAxis0_fPhi is None: tAxis0_fPhi = thrust.getThrustAxis(evt0_fPhi)
        dPhiThrust_fPhi = tAxis0_fPhi.DeltaPhi(tAxis1) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fPhi,-dPhiThrust_fPhi),
                               ev1,R,periodic_phi=periodic_phi))
        dPhiThrust_fPhi_pi = (tAxis0_fPhi.DeltaPhi(tAxis1) + np.pi) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fPhi,-dPhiThrust_fPhi_pi),
                               ev1,R,periodic_phi=periodic_phi))
        
    else:
        evt0_fEta = flipEta(ev0)
        if tAxis0_fEta is None: tAxis0_fEta = thrust.getThrustAxis(evt0_fEta)

        dPhiThrust_fEta = tAxis0_fEta.DeltaPhi(tAxis1) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fEta,-dPhiThrust_fEta),
                               ev1,R,periodic_phi=periodic_phi))
        dPhiThrust_fEta_pi = (tAxis0_fEta.DeltaPhi(tAxis1) + np.pi) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fEta,-dPhiThrust_fEta_pi),
                               ev1,R,periodic_phi=periodic_phi))
        
        evt0_fPhifEta   = flipPhi(flipEta(ev0))
        if tAxis0_fPhifEta is None: tAxis0_fPhifEta = thrust.getThrustAxis(evt0_fPhifEta)
        dPhiThrust_fPhifEta = tAxis0_fPhifEta.DeltaPhi(tAxis1) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fPhifEta,-dPhiThrust_fPhifEta),
                               ev1,R,periodic_phi=periodic_phi))
        dPhiThrust_fPhifEta_pi = (tAxis0_fPhifEta.DeltaPhi(tAxis1) + np.pi) % (2*np.pi)
        thrustCalcs.append(emd(movePhi(evt0_fPhifEta,-dPhiThrust_fPhifEta_pi),
                               ev1,R,periodic_phi=periodic_phi))


    #
    #  Get the min
    #
    emdval = np.min(thrustCalcs)
        

    if return_flow:
        

        #
        #  Get Eta Right
        #
        event0_etaCorrect = ev0
        if doEtaFlip:
            event0_etaCorrect = flipEta(ev0)

        #
        #  Get Phi Right
        #        
        minIndx = np.argmin(thrustCalcs)
        if minIndx in [0,1]:
            event0_etaAndPhiCorrect = event0_etaCorrect
        else:
            event0_etaAndPhiCorrect = flipPhi(event0_etaCorrect)
            
        #
        #  Get Correct Delta Phi
        #
        tAxis0_min = thrust.getThrustAxis(event0_etaAndPhiCorrect)
        dPhiThrust_min = tAxis0_min.DeltaPhi(tAxis1) % (2*np.pi)
        if minIndx in [1,3]:
            dPhiThrust_min = (tAxis0_min.DeltaPhi(tAxis1) + np.pi) % (2*np.pi) 
        evt0_min = movePhi(event0_etaAndPhiCorrect,dPhiThrust_min)

        emdVal, G = emd(evt0_min,ev1,R,return_flow=True,periodic_phi=periodic_phi)
        return  emdVal, G ,evt0_min

    return emdval





