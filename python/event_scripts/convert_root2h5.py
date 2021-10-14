import pandas as pd
import ROOT
ROOT.gROOT.SetBatch(True)
import numpy as np
import os, sys
from glob import glob
from copy import copy
import multiprocessing
mZ, mH = 91.0, 125.0
import argparse

sys.path.insert(0, ".")

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inFile', default='../../events/MG2/TTree/bbbj.root', type=str, help='Input root file.')
parser.add_argument('-o', '--outfile', default='../fvt_scripts/bbbj.h5', type=str, help='Output pq file dir. Default is input file name with .root->.h5')
parser.add_argument('-d', '--debug', dest="debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

inPaths = args.inFile.split()
inFiles = []
for path in inPaths:
    inFiles += glob(path)
print(inFiles)


class variable:
    def __init__(self, name, dtype=np.float32):
        self.name = name
        self.status = False
        self.dtype = dtype

    def setStatus(self,tree):
        self.status = 0 if 'nil' in str(tree.FindBranch(self.name)) else 1
        print(self.name, self.status)
        if self.status: tree.SetBranchStatus(self.name,self.status)

variables = [variable("FvT"),
             variable("FvT_pd4"),
             variable("FvT_pd3"),
             variable("FvT_pt4"),
             variable("FvT_pt3"),
             variable("FvT_pm4"),
             variable("FvT_pm3"),
             variable("HLT_PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5", dtype=np.bool_),
             variable("nIsoMuons",dtype=np.uint32),
             variable("xWbW"),
             variable("xbW"),
             variable("xW"),
             variable("xWt"),
             variable("xt"),
             variable("m4j"),
             variable("SB", dtype=np.bool_), variable("CR", dtype=np.bool_), variable("SR", dtype=np.bool_),
             variable("ZZSB", dtype=np.bool_), variable("ZZCR", dtype=np.bool_), variable("ZZSR", dtype=np.bool_),             
             variable("ZHSB", dtype=np.bool_), variable("ZHCR", dtype=np.bool_), variable("ZHSR", dtype=np.bool_),             
             variable("passXWt", dtype=np.bool_), variable("passHLT", np.bool_),
             variable("weight"),
             variable("pseudoTagWeight"),
             variable("mcPseudoTagWeight"),
             variable("fourTag", dtype=np.bool_),
             variable("dRjjClose"), variable("dRjjOther"),
             variable("nSelJets", dtype=np.uint32),
             variable("nPSTJets", dtype=np.uint32),
             variable("st"),
             variable("stNotCan"),
             variable("aveAbsEta"),
             variable("aveAbsEtaOth"),
             variable("nPVsGood"),
             ]

def convert(inFile):

    tree = ROOT.TChain("Tree")
    tree.Add(inFile)

    tree.Show(0)

    # Initialize TTree

    print("!")
    tree.SetBranchStatus("*",0)
    tree.SetBranchStatus("jetPt",1); 
    print("#")
##    tree.SetBranchStatus("jetPt[1]",1); 
##    tree.SetBranchStatus("jetPt[2]",1); 
##    tree.SetBranchStatus("jetPt[3]",1)

    tree.SetBranchStatus("jetEta",1); 
##    tree.SetBranchStatus("jetEta[1]",1); 
##    tree.SetBranchStatus("jetEta[2]",1); 
##    tree.SetBranchStatus("jetEta[3]",1)

    tree.SetBranchStatus("jetPhi",1); 
##    tree.SetBranchStatus("jetPhi[1]",1); 
##    tree.SetBranchStatus("jetPhi[2]",1); 
##    tree.SetBranchStatus("jetPhi[3]",1)

##    canJet_m_Status = False if "nil" in str(tree.FindBranch("canJet0_m")) else True

    canJet_m_Status = False

    if canJet_m_Status:
        tree.SetBranchStatus("canJet0_m",1); 
        tree.SetBranchStatus("canJet1_m",1); 
        tree.SetBranchStatus("canJet2_m",1); 
        tree.SetBranchStatus("canJet3_m",1)

    else:
        tree.SetBranchStatus("jetEnergy",1); 
##        tree.SetBranchStatus("jetEnergy[1]",1); 
##        tree.SetBranchStatus("jetEnergy[2]",1); 
##        tree.SetBranchStatus("jetEnergy[3]",1)

##    tree.SetBranchStatus("nAllNotCanJets",1)
##    tree.SetBranchStatus("notCanJet_pt",1)
##    tree.SetBranchStatus("notCanJet_eta",1)
##    tree.SetBranchStatus("notCanJet_phi",1)
##    tree.SetBranchStatus("notCanJet_m",1)

    print("$$$$")

    for var in variables:
        print(var.name)
        var.setStatus(tree)
    print("@@#@")
    tree.Show(0)

    nEvts = tree.GetEntries()
    assert nEvts > 0
    print(" >> Input file:",inFile)
    print(" >> nEvts:",nEvts)
    outfile = args.outfile if args.outfile else inFile.replace(".root",".h5")
    print(" >> Output file:",outfile)
    store = pd.HDFStore(outfile,mode='w')
    #store.close()
    print("%%")
    ##### Start Conversion #####

    # Event range to process
    iEvtStart = 0
    iEvtEnd   = 10000
    iEvtEnd   = nEvts 
    chunkSize = 10000
    assert iEvtEnd <= nEvts
    print(" >> Processing entries: [",iEvtStart,"->",iEvtEnd,")")

    sw = ROOT.TStopwatch()
    sw.Start()

    nWritten = 0
    for iEvtStart in range(0,iEvtEnd,chunkSize):
        data = {'canJet0_pt' : [], 'canJet1_pt' : [], 'canJet2_pt' : [], 'canJet3_pt' : [],
                'canJet0_eta': [], 'canJet1_eta': [], 'canJet2_eta': [], 'canJet3_eta': [],
                'canJet0_phi': [], 'canJet1_phi': [], 'canJet2_phi': [], 'canJet3_phi': [],
                'canJet0_m'  : [], 'canJet1_m'  : [], 'canJet2_m'  : [], 'canJet3_m'  : [],
                #'canJet0_e'  : [], 'canJet1_e'  : [], 'canJet2_e'  : [], 'canJet3_e'  : [],
                'm01': [], 'm23': [], 'm02': [], 'm13': [], 'm03': [], 'm12': [], 
                'm123': [], 'm023': [], 'm013': [], 'm012': [],
                'pt01': [], 'pt23': [], 'pt02': [], 'pt13': [], 'pt03': [], 'pt12': [], 
                'dR01': [], 'dR23': [], 'dR02': [], 'dR13': [], 'dR03': [], 'dR12': [], 
                's4j': [],
                'dR0123': [], 'dR0213': [], 'dR0312': [],
                } 
        nOthJetsMax = 12
        for i in range(nOthJetsMax):
            data['notCanJet'+str(i)+'_pt'] = []
            data['notCanJet'+str(i)+'_eta'] = []
            data['notCanJet'+str(i)+'_phi'] = []
            data['notCanJet'+str(i)+'_m'] = []
            data['notCanJet'+str(i)+'_isSelJet'] = []

        for var in variables:
            if var.status: data[var.name] = []

        for iEvt in range(iEvtStart, min(iEvtStart+chunkSize, iEvtEnd)):

            # Initialize event
            tree.GetEntry(iEvt)
            if (iEvt+1) % 1000 == 0 or iEvt+1 == iEvtEnd:
                sys.stdout.write("\rProcessed "+str(iEvt+1)+" of "+str(nEvts)+" | "+str(int((iEvt+1)*100.0/nEvts))+"% ")
                sys.stdout.flush()


            jets = [ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector(),ROOT.TLorentzVector()]
            print("@@")
            data['canJet0_pt'].append(copy(tree.jetPt[0])); 
            print("$$")
            data['canJet1_pt'].append(copy(tree.canJet1_pt)); 
            data['canJet2_pt'].append(copy(tree.canJet2_pt)); 
            data['canJet3_pt'].append(copy(tree.canJet3_pt))
            data['canJet0_eta'].append(copy(tree.canJet0_eta)); 
            data['canJet1_eta'].append(copy(tree.canJet1_eta)); 
            data['canJet2_eta'].append(copy(tree.canJet2_eta)); 
            data['canJet3_eta'].append(copy(tree.canJet3_eta))
            data['canJet0_phi'].append(copy(tree.canJet0_phi)); 
            data['canJet1_phi'].append(copy(tree.canJet1_phi)); 
            data['canJet2_phi'].append(copy(tree.canJet2_phi)); 
            data['canJet3_phi'].append(copy(tree.canJet3_phi))

            if canJet_m_Status:
                data['canJet0_m'].append(copy(tree.canJet0_m)); data['canJet1_m'].append(copy(tree.canJet1_m)); data['canJet2_m'].append(copy(tree.canJet2_m)); data['canJet3_m'].append(copy(tree.canJet3_m))
                jets[0].SetPtEtaPhiM(tree.canJet0_pt, tree.canJet0_eta, tree.canJet0_phi, tree.canJet0_m)
                jets[1].SetPtEtaPhiM(tree.canJet1_pt, tree.canJet1_eta, tree.canJet1_phi, tree.canJet1_m)
                jets[2].SetPtEtaPhiM(tree.canJet2_pt, tree.canJet2_eta, tree.canJet2_phi, tree.canJet2_m)
                jets[3].SetPtEtaPhiM(tree.canJet3_pt, tree.canJet3_eta, tree.canJet3_phi, tree.canJet3_m)

            else:
                # data['canJet0_e'].append(copy(tree.canJet0_e)); data['canJet1_e'].append(copy(tree.canJet1_e)); data['canJet2_e'].append(copy(tree.canJet2_e)); data['canJet3_e'].append(copy(tree.canJet3_e))
                jets[0].SetPtEtaPhiE(tree.canJet0_pt, tree.canJet0_eta, tree.canJet0_phi, tree.canJet0_e)
                jets[1].SetPtEtaPhiE(tree.canJet1_pt, tree.canJet1_eta, tree.canJet1_phi, tree.canJet1_e)
                jets[2].SetPtEtaPhiE(tree.canJet2_pt, tree.canJet2_eta, tree.canJet2_phi, tree.canJet2_e)
                jets[3].SetPtEtaPhiE(tree.canJet3_pt, tree.canJet3_eta, tree.canJet3_phi, tree.canJet3_e)
                data['canJet0_m'].append(jets[0].M()); data['canJet1_m'].append(jets[1].M()); data['canJet2_m'].append(jets[2].M()); data['canJet3_m'].append(jets[3].M())



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

            m123 = (jets[1]+jets[2]+jets[3]).M() #missing 0
            m023 = (jets[0]+jets[2]+jets[3]).M() #missing 1
            m013 = (jets[0]+jets[1]+jets[3]).M() #missing 2
            m012 = (jets[0]+jets[1]+jets[2]).M() #missing 3
            data['m123'].append(m123)
            data['m023'].append(m023)
            data['m013'].append(m013)
            data['m012'].append(m012)

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

            data['s4j'].append(tree.canJet0_pt + tree.canJet1_pt + tree.canJet2_pt + tree.canJet3_pt)

            for i in range(nOthJetsMax):
                if i < tree.nAllNotCanJets:
                    data['notCanJet'+str(i)+'_pt'].append(copy(tree.notCanJet_pt[i]))
                    data['notCanJet'+str(i)+'_eta'].append(copy(tree.notCanJet_eta[i]))
                    data['notCanJet'+str(i)+'_phi'].append(copy(tree.notCanJet_phi[i]))
                    data['notCanJet'+str(i)+'_m'].append(copy(tree.notCanJet_m[i]))
                    data['notCanJet'+str(i)+'_isSelJet'].append(1 if tree.notCanJet_pt[i]>40 and abs(tree.notCanJet_eta[i])<2.4 else 0)
                    if abs(tree.notCanJet_eta[i])>2.4 and tree.notCanJet_pt[i]<40: print("ERROR: This notCanJet should have failed forward pileup veto",tree.notCanJet_eta[i],tree.notCanJet_pt[i])
                else:
                    data['notCanJet'+str(i)+'_pt'].append(0)
                    data['notCanJet'+str(i)+'_eta'].append(0)
                    data['notCanJet'+str(i)+'_phi'].append(0)
                    data['notCanJet'+str(i)+'_m'].append(0)
                    data['notCanJet'+str(i)+'_isSelJet'].append(-1)

            for var in variables:
                if var.status: data[var.name].append(copy(getattr(tree, var.name)))

            nWritten += 1

        #print

        data['s4j'] = np.array(data['s4j'], np.float32)
        data['canJet0_pt'] = np.array(data['canJet0_pt'], np.float32); data['canJet1_pt'] = np.array(data['canJet1_pt'], np.float32); data['canJet2_pt'] = np.array(data['canJet2_pt'], np.float32); data['canJet3_pt'] = np.array(data['canJet3_pt'], np.float32)
        data['canJet0_eta'] = np.array(data['canJet0_eta'], np.float32); data['canJet1_eta'] = np.array(data['canJet1_eta'], np.float32); data['canJet2_eta'] = np.array(data['canJet2_eta'], np.float32); data['canJet3_eta'] = np.array(data['canJet3_eta'], np.float32)
        data['canJet0_phi'] = np.array(data['canJet0_phi'], np.float32); data['canJet1_phi'] = np.array(data['canJet1_phi'], np.float32); data['canJet2_phi'] = np.array(data['canJet2_phi'], np.float32); data['canJet3_phi'] = np.array(data['canJet3_phi'], np.float32)
        data['canJet0_m'] = np.array(data['canJet0_m'], np.float32); data['canJet1_m'] = np.array(data['canJet1_m'], np.float32); data['canJet2_m'] = np.array(data['canJet2_m'], np.float32); data['canJet3_m'] = np.array(data['canJet3_m'], np.float32)
        data['m01'] = np.array(data['m01'], np.float32)
        data['m23'] = np.array(data['m23'], np.float32)
        data['m02'] = np.array(data['m02'], np.float32)
        data['m13'] = np.array(data['m13'], np.float32)
        data['m03'] = np.array(data['m03'], np.float32)
        data['m12'] = np.array(data['m12'], np.float32)
        data['m123'] = np.array(data['m123'], np.float32)
        data['m023'] = np.array(data['m023'], np.float32)
        data['m013'] = np.array(data['m013'], np.float32)
        data['m012'] = np.array(data['m012'], np.float32)
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

        for i in range(nOthJetsMax):
            data['notCanJet'+str(i)+'_pt'] = np.array(data['notCanJet'+str(i)+'_pt'], np.float32)
            data['notCanJet'+str(i)+'_eta'] = np.array(data['notCanJet'+str(i)+'_eta'], np.float32)
            data['notCanJet'+str(i)+'_phi'] = np.array(data['notCanJet'+str(i)+'_phi'], np.float32)
            data['notCanJet'+str(i)+'_m'] = np.array(data['notCanJet'+str(i)+'_m'], np.float32)
            data['notCanJet'+str(i)+'_isSelJet'] = np.array(data['notCanJet'+str(i)+'_isSelJet'], np.float32)

        for var in variables:
            if var.status: data[var.name] = np.array(data[var.name], var.dtype)

        df=pd.DataFrame(data)
        store.append('df', df, format='table', data_columns=None, index=False)
        del df
        del data

    store.close()
    sw.Stop()
    print
    print( " >> nWritten:",nWritten)
    print( " >> Real time:",sw.RealTime()/60.,"minutes")
    print( " >> CPU time: ",sw.CpuTime() /60.,"minutes")
    print( " >> ======================================")



workers = multiprocessing.Pool(12)
done=0
for output in workers.imap_unordered(convert,inFiles):
    if output != None:
        print(output)
    else: 
        done+=1
        print("finished converting",done,"of",len(inFiles),"files")

for f in inFiles: print("converted:",f)
print("done")
