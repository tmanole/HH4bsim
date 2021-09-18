import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy
mZ, mH = 91.0, 125.0

print("start")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--infile', default='/home/Documents/CMU/Research/LHC/code/toy4b/events/ROOT16d/bbbb_run_02_toyTree.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='.', type=str, help='Output pq file dir. Default is input file name with .root->.h5')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
parser.add_argument('-t', '--tags', default=1, type=int, help='Tree has four tag events?')
args = parser.parse_args()

fourTag = False if args.tags == 0 else True

print(fourTag)

inStr=args.infile
inFiles = glob(inStr)



for inFile in inFiles:
    #bbbb_tree.SetName("t4b")

    #tfile = ROOT.TFile()
    #tree = tfile.Get("Tree")

    

    tree = ROOT.TChain("Tree")
    tree.Add(inFile)

    # Initialize TTree
    tree.SetBranchStatus("*",0)
    tree.SetBranchStatus("nJets",1)
    tree.SetBranchStatus("jetPt",1)#; tree.SetBranchStatus("jetPt[1]",1); tree.SetBranchStatus("jetPt[2]",1); tree.SetBranchStatus("jetPt[3]",1)
    tree.SetBranchStatus("jetEta",1)#; tree.SetBranchStatus("jetEta[1]",1); tree.SetBranchStatus("jetEta[2]",1); tree.SetBranchStatus("jetEta[3]",1)
    tree.SetBranchStatus("jetPhi",1)#; tree.SetBranchStatus("jetPhi[1]",1); tree.SetBranchStatus("jetPhi[2]",1); tree.SetBranchStatus("jetPhi[3]",1)
    tree.SetBranchStatus("jetEnergy",1)#; tree.SetBranchStatus("jetEnergy[1]",1); tree.SetBranchStatus("jetEnergy[2]",1); tree.SetBranchStatus("jetEnergy[3]",1)
    tree.SetBranchStatus("dRjjClose",1)
    tree.SetBranchStatus("dRjjOther",1)
    tree.SetBranchStatus("aveAbsEta",1)
    tree.SetBranchStatus("m4j",1)
    tree.SetBranchStatus("mZZ",1)
    tree.SetBranchStatus("SB",1); tree.SetBranchStatus("CR",1), tree.SetBranchStatus("SR",1)
    tree.SetBranchStatus("weight",1)

    tree.Show(0)

    nEvts = tree.GetEntries()
    assert nEvts > 0
    print(" >> Input file:",inFile)
    print(" >> nEvts:",nEvts)
    outfile = args.outfile if args.outfile else inFile.replace(".root",".h5")
    print(" >> Output file:", outfile)
    
    ##### Start Conversion #####

    # Event range to process
    iEvtStart = 0
    iEvtEnd   = 1000
    iEvtEnd   = nEvts 
    assert iEvtEnd <= nEvts
    print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

    nWritten = 0
    data = {'canJet0_pt' : [], 'canJet1_pt' : [], 'canJet2_pt' : [], 'canJet3_pt' : [],
            'canJet0_eta': [], 'canJet1_eta': [], 'canJet2_eta': [], 'canJet3_eta': [],
            'canJet0_phi': [], 'canJet1_phi': [], 'canJet2_phi': [], 'canJet3_phi': [],
            'canJet0_e'  : [], 'canJet1_e'  : [], 'canJet2_e'  : [], 'canJet3_e'  : [],
            'canJet0_m'  : [], 'canJet1_m'  : [], 'canJet2_m'  : [], 'canJet3_m'  : [],
            'dRjjClose': [],
            'dRjjOther': [],
            'aveAbsEta': [],
            'm4j': [],
            'SB': [], 
            'CR': [], 
            'SR': [],
            'weight': [],
            'fourTag': [],  
            'm01': [], 'm23': [], 'm02': [], 'm13': [], 'm03': [], 'm12': [], 
            'pt01': [], 'pt23': [], 'pt02': [], 'pt13': [], 'pt03': [], 'pt12': [], 
            'dR01': [], 'dR23': [], 'dR02': [], 'dR13': [], 'dR03': [], 'dR12': [], 
            's4j': [],
            'dR0123': [], 'dR0213': [], 'dR0312': [],
            'mZZ0123': [], 'mZZ0213': [], 'mZZ0312': []}

    sw = ROOT.TStopwatch()
    sw.Start()
    for iEvt in list(range(iEvtStart,iEvtEnd)):

        # Initialize event
        tree.GetEntry(iEvt)
        if (iEvt+1) % 1000 == 0 or iEvt+1 == iEvtEnd:
            sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
            sys.stdout.flush()


        data['canJet0_pt'].append(copy(tree.jetPt[0])) 
        data['canJet1_pt'].append(copy(tree.jetPt[1])) 
        data['canJet2_pt'].append(copy(tree.jetPt[2])) 
        data['canJet3_pt'].append(copy(tree.jetPt[3]))
        data['canJet0_eta'].append(copy(tree.jetEta[0])) 
        data['canJet1_eta'].append(copy(tree.jetEta[1])) 
        data['canJet2_eta'].append(copy(tree.jetEta[2])) 
        data['canJet3_eta'].append(copy(tree.jetEta[3]))
        data['canJet0_phi'].append(copy(tree.jetPhi[0]))
        data['canJet1_phi'].append(copy(tree.jetPhi[1]))
        data['canJet2_phi'].append(copy(tree.jetPhi[2]))
        data['canJet3_phi'].append(copy(tree.jetPhi[3]))
        data['canJet0_e'].append(copy(tree.jetEnergy[0]))
        data['canJet1_e'].append(copy(tree.jetEnergy[1]))
        data['canJet2_e'].append(copy(tree.jetEnergy[2]))
        data['canJet3_e'].append(copy(tree.jetEnergy[3]))

        jets = [ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector(), ROOT.TLorentzVector()]
        jets[0].SetPtEtaPhiE(tree.jetPt[0], tree.jetEta[0], tree.jetPhi[0], tree.jetEnergy[0])
        jets[1].SetPtEtaPhiE(tree.jetPt[1], tree.jetEta[1], tree.jetPhi[1], tree.jetEnergy[1])
        jets[2].SetPtEtaPhiE(tree.jetPt[2], tree.jetEta[2], tree.jetPhi[2], tree.jetEnergy[2])
        jets[3].SetPtEtaPhiE(tree.jetPt[3], tree.jetEta[3], tree.jetPhi[3], tree.jetEnergy[3])
        data['canJet0_m'].append(copy(jets[0].M()))
        data['canJet1_m'].append(copy(jets[1].M()))
        data['canJet2_m'].append(copy(jets[2].M()))
        data['canJet3_m'].append(copy(jets[3].M()))

        d01, d23 = jets[0]+jets[1], jets[2]+jets[3]
        d02, d13 = jets[0]+jets[2], jets[1]+jets[3]
        d03, d12 = jets[0]+jets[3], jets[1]+jets[2]

        m01, m23 = d01.M(), d23.M()
        m02, m13 = d02.M(), d13.M()
        m03, m12 = d03.M(), d12.M()
        data['m01'].append(m01)
        data['m23'].append(m23)
        data['m02'].append(m02)
        data['m13'].append(m13)
        data['m03'].append(m03)
        data['m12'].append(m12)

        pt01, pt23 = d01.Pt(), d23.Pt()
        pt02, pt13 = d02.Pt(), d13.Pt()
        pt03, pt12 = d03.Pt(), d12.Pt()
        data['pt01'].append(pt01)
        data['pt23'].append(pt23)
        data['pt02'].append(pt02)
        data['pt13'].append(pt13)
        data['pt03'].append(pt03)
        data['pt12'].append(pt12)

        dR01 = jets[0].DeltaR(jets[1])
        dR23 = jets[2].DeltaR(jets[3])
        dR02 = jets[0].DeltaR(jets[2])
        dR13 = jets[1].DeltaR(jets[3])
        dR03 = jets[0].DeltaR(jets[3])
        dR12 = jets[1].DeltaR(jets[2])
        data['dR01'].append(dR01)
        data['dR23'].append(dR23)
        data['dR02'].append(dR02)
        data['dR13'].append(dR13)
        data['dR03'].append(dR03)
        data['dR12'].append(dR12)
    
        dR0123 = d01.DeltaR(d23)
        dR0213 = d02.DeltaR(d13)
        dR0312 = d03.DeltaR(d12)
        data['dR0123'].append(dR0123)
        data['dR0213'].append(dR0213)
        data['dR0312'].append(dR0312)

        # ZH code.
        ds0123 = [d01, d23] if m01 > m23 else [d23, d01]
        ds0213 = [d02, d13] if m02 > m13 else [d13, d02]
        ds0312 = [d03, d12] if m03 > m12 else [d12, d03]
        #mZH0123 = (ds0123[0]*(mH/ds0123[0].M()) + ds0123[1]*(mZ/ds0123[1].M())).M()
        #mZH0213 = (ds0213[0]*(mH/ds0213[0].M()) + ds0213[1]*(mZ/ds0213[1].M())).M()
        #mZH0312 = (ds0312[0]*(mH/ds0312[0].M()) + ds0312[1]*(mZ/ds0312[1].M())).M()
        #data['mZH0123'].append(mZH0123)
        #data['mZH0213'].append(mZH0213)
        #data['mZH0312'].append(mZH0312)

        mZZ0123 = (ds0123[0]*(mZ/ds0123[0].M()) + ds0123[1]*(mZ/ds0123[1].M())).M()
        mZZ0213 = (ds0213[0]*(mZ/ds0213[0].M()) + ds0213[1]*(mZ/ds0213[1].M())).M()
        mZZ0312 = (ds0312[0]*(mZ/ds0312[0].M()) + ds0312[1]*(mZ/ds0312[1].M())).M()
        data['mZZ0123'].append(mZZ0123)
        data['mZZ0213'].append(mZZ0213)
        data['mZZ0312'].append(mZZ0312)
    
        #data['st'].append(copy(tree.st))
        #data['stNotCan'].append(copy(tree.stNotCan))
        data['s4j'].append(tree.jetPt[0] + tree.jetPt[1] + tree.jetPt[2] + tree.jetPt[3])
        data['m4j'].append(copy(tree.m4j))
        #data['xWt0'].append(copy(tree.xWt0))
        #data['xWt1'].append(copy(tree.xWt1))
        data['weight']    .append(copy(tree.weight))
        #data['nPVsGood']    .append(copy(tree.nPVsGood))
        #data['pseudoTagWeight']    .append(copy(tree.pseudoTagWeight))
        #data['passHLT'].append(copy(tree.passHLT))
        #data['ZHSB'].append(copy(tree.ZHSB)); data['ZHCR'].append(copy(tree.ZHCR)); data['ZHSR'].append(copy(tree.ZHSR))
        #data['ZZSB'].append(copy(tree.ZZSB)); data['ZZCR'].append(copy(tree.ZZCR)); data['ZZSR'].append(copy(tree.ZZSR))
        data['SB'].append(copy(tree.SB)); data['CR'].append(copy(tree.CR)); data['SR'].append(copy(tree.SR))
        #data['passDEtaBB'].append(copy(tree.passDEtaBB))
        data['fourTag']   .append(fourTag) 
        #data['nSelJets'].append(copy(tree.nSelJets))
        #data['nPSTJets'].append(copy(tree.nPSTJets))
        data['dRjjClose'] .append(copy(tree.dRjjClose))
        data['dRjjOther'] .append(copy(tree.dRjjOther))
        data['aveAbsEta'] .append(copy(tree.aveAbsEta))
        #data['aveAbsEtaOth'] .append(copy(tree.aveAbsEtaOth))

    #print

    #data['st'] = np.array(data['st'], np.float32)
    #data['stNotCan'] = np.array(data['stNotCan'], np.float32)
    data['s4j'] = np.array(data['s4j'], np.float32)
    data['m4j'] = np.array(data['m4j'], np.float32)
    #data['xWt0'] = np.array(data['xWt0'], np.float32)
    #data['xWt1'] = np.array(data['xWt1'], np.float32)
    data['weight']     = np.array(data['weight'],     np.float32)
    #data['nPVsGood']     = np.array(data['nPVsGood'],     np.float32)
    #data['pseudoTagWeight']     = np.array(data['pseudoTagWeight'],     np.float32)
    #data['ZHSB'] = np.array(data['ZHSB'], np.bool_); data['ZHCR'] = np.array(data['ZHCR'], np.bool_); data['ZHSR'] = np.array(data['ZHSR'], np.bool_)
    #data['ZZSB'] = np.array(data['ZZSB'], np.bool_); data['ZZCR'] = np.array(data['ZZCR'], np.bool_); data['ZZSR'] = np.array(data['ZZSR'], np.bool_)
    data['SB'] = np.array(data['SB'], np.bool_); data['CR'] = np.array(data['CR'], np.bool_); data['SR'] = np.array(data['SR'], np.bool_)
    #data['passHLT'] = np.array(data['passHLT'], np.bool_)
    #data['passDEtaBB'] = np.array(data['passDEtaBB'], np.bool_)
    data['fourTag']    = np.array(data['fourTag'],    np.bool_)
    data['canJet0_pt'] = np.array(data['canJet0_pt'], np.float32)
    data['canJet1_pt'] = np.array(data['canJet1_pt'], np.float32)
    data['canJet2_pt'] = np.array(data['canJet2_pt'], np.float32)
    data['canJet3_pt'] = np.array(data['canJet3_pt'], np.float32)
    data['canJet0_eta'] = np.array(data['canJet0_eta'], np.float32)
    data['canJet1_eta'] = np.array(data['canJet1_eta'], np.float32)
    data['canJet2_eta'] = np.array(data['canJet2_eta'], np.float32)
    data['canJet3_eta'] = np.array(data['canJet3_eta'], np.float32)
    data['canJet0_phi'] = np.array(data['canJet0_phi'], np.float32)
    data['canJet1_phi'] = np.array(data['canJet1_phi'], np.float32)
    data['canJet2_phi'] = np.array(data['canJet2_phi'], np.float32)
    data['canJet3_phi'] = np.array(data['canJet3_phi'], np.float32)
    data['canJet0_e'] = np.array(data['canJet0_e'], np.float32)
    data['canJet1_e'] = np.array(data['canJet1_e'], np.float32)
    data['canJet2_e'] = np.array(data['canJet2_e'], np.float32)
    data['canJet3_e'] = np.array(data['canJet3_e'], np.float32)
    data['canJet0_m'] = np.array(data['canJet0_m'], np.float32)
    data['canJet1_m'] = np.array(data['canJet1_m'], np.float32)
    data['canJet2_m'] = np.array(data['canJet2_m'], np.float32)
    data['canJet3_m'] = np.array(data['canJet3_m'], np.float32)
    data['m01'] = np.array(data['m01'], np.float32)
    data['m23'] = np.array(data['m23'], np.float32)
    data['m02'] = np.array(data['m02'], np.float32)
    data['m13'] = np.array(data['m13'], np.float32)
    data['m03'] = np.array(data['m03'], np.float32)
    data['m12'] = np.array(data['m12'], np.float32)
    data['pt01'] = np.array(data['pt01'], np.float32)
    data['pt23'] = np.array(data['pt23'], np.float32)
    data['pt02'] = np.array(data['pt02'], np.float32)
    data['pt13'] = np.array(data['pt13'], np.float32)
    data['pt03'] = np.array(data['pt03'], np.float32)
    data['pt12'] = np.array(data['pt12'], np.float32)
    data['dR01'] = np.array(data['dR01'], np.float32)
    data['dR23'] = np.array(data['dR23'], np.float32)
    data['dR02'] = np.array(data['dR02'], np.float32)
    data['dR13'] = np.array(data['dR13'], np.float32)
    data['dR03'] = np.array(data['dR03'], np.float32)
    data['dR12'] = np.array(data['dR12'], np.float32)
    data['dR0123'] = np.array(data['dR0123'], np.float32)
    data['dR0213'] = np.array(data['dR0213'], np.float32)
    data['dR0312'] = np.array(data['dR0312'], np.float32)
    #data['mZH0123'] = np.array(data['mZH0123'], np.float32)
    #data['mZH0213'] = np.array(data['mZH0213'], np.float32)
    #data['mZH0312'] = np.array(data['mZH0312'], np.float32)
    data['mZZ0123'] = np.array(data['mZZ0123'], np.float32)
    data['mZZ0213'] = np.array(data['mZZ0213'], np.float32)
    data['mZZ0312'] = np.array(data['mZZ0312'], np.float32)
    #data['nSelJets']   = np.array(data['nSelJets'],   np.uint32)
    #data['nPSTJets']   = np.array(data['nPSTJets'],   np.uint32)
    data['dRjjClose']  = np.array(data['dRjjClose'],  np.float32)
    data['dRjjOther']  = np.array(data['dRjjOther'],  np.float32)
    data['aveAbsEta']  = np.array(data['aveAbsEta'],  np.float32)
    #data['aveAbsEtaOth']  = np.array(data['aveAbsEtaOth'],  np.float32)

    #if FvTStatus: data['FvT'] = np.array(data['FvT'], np.float32)
    #if ZHvBStatus: data['ZHvB'] = np.array(data['ZHvB'], np.float32)
    #if ZZvBStatus: data['ZZvB'] = np.array(data['ZZvB'], np.float32)

    for key,value in data.items():
        print(key)
        print(value.shape)

    df=pd.DataFrame(data)
    print("df.dtypes")
    print(df.dtypes)
    print("df.shape", df.shape)

    df.to_hdf(outfile, key='df', format='table', mode='w')

    sw.Stop()
    print("\n")
    print(" >> nWritten:", nWritten)
    print(" >> Real time:", sw.RealTime()/60.,"minutes")
    print(" >> CPU time: ", sw.CpuTime() /60.,"minutes")
    print(" >> ======================================")

