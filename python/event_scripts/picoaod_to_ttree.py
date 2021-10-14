# converts real data events from the format in the picoAOD files to the format of the simulated data so that
# the same methods can be used on both. Note that this only generates the TTree files
import ROOT
import numpy as np
from array import array
file_path = '../../events/PicoAODs/data2016'
outpath = '../../events/RealAll/TTree/bbbb'
outpath2 = '../../events/Real/TTree/bbbb'
outpath3 = '../../events/Real5/TTree/bbbb'

#build a class like the one used to make the simulated TTrees
class ToyTree:
    # init copied from Patrick's ZZ4b repo
    def __init__(self, name, debug = False):
        self.name = name
        self.debug = debug
        self.f = ROOT.TFile(name+".root","RECREATE")
        self.t = ROOT.TTree("Tree",name)
        #Jet 4-vectors
        maxJets = 20
        self.n   = array('i',         [0])
        self.pt  = array('f', maxJets*[0])
        self.eta = array('f', maxJets*[0])
        self.phi = array('f', maxJets*[0])
        self.e   = array('f', maxJets*[0])
        self.t.Branch('nJets',     self.n,   'nJets/I')
        self.t.Branch('jetPt',     self.pt,  'jetPt[nJets]/F')
        self.t.Branch('jetEta',    self.eta, 'jetEta[nJets]/F')
        self.t.Branch('jetPhi',    self.phi, 'jetPhi[nJets]/F')
        self.t.Branch('jetEnergy', self.e,   'jetEnergy[nJets]/F')
        #High Level Variables
        self.dRjjClose = array('f', [0])
        self.dRjjOther = array('f', [0])
        self.aveAbsEta = array('f', [0])
        self.t.Branch('dRjjClose', self.dRjjClose, 'dRjjClose/F')
        self.t.Branch('dRjjOther', self.dRjjOther, 'dRjjOther/F')
        self.t.Branch('aveAbsEta', self.aveAbsEta, 'aveAbsEta/F')
        self.m4j = array('f', [0])
        self.mHH = array('f', [0])
        self.t.Branch('m4j', self.m4j, 'm4j/F')
        self.t.Branch('mHH', self.mHH, 'mHH/F')
        #Region
        self.SB = array('i', [0])
        self.CR = array('i', [0])
        self.SR = array('i', [0])
        self.t.Branch('SB', self.SB, 'SB/I')
        self.t.Branch('CR', self.CR, 'CR/I')
        self.t.Branch('SR', self.SR, 'SR/I')
        #Weight
        self.weight = array('f', [1])
        self.t.Branch('weight', self.weight, 'weight/F')

    ### Edited to use event_data array instead of an event class
    def Fill(self, event_data):
        self.n[0] = 4
        for i in list(range(4)):
            self.pt [i] = event_data[1][i]
            self.eta[i] = event_data[2][i]
            self.phi[i] = event_data[3][i]
            self.e  [i] = event_data[4][i]
        self.dRjjClose[0] = event_data[5]
        self.dRjjOther[0] = event_data[6]
        self.aveAbsEta[0] = event_data[7]
        self.m4j[0] = event_data[8]
        self.mHH[0] = event_data[9]
        self.SB[0] = event_data[10]
        self.CR[0] = event_data[11]
        self.SR[0] = event_data[12]
        self.weight[0] = 1.0
        self.t.Fill()

    # write copied from ZZ4b
    def Write(self):
        print(self.name+".root:",self.t.GetEntries()," entries")
        if self.debug: self.t.Show(0)
        self.f.Write()
        self.f.Close()
        
# gathered how to do this from eventViews, eventData, analysis in ZZ4b
def get_mHH(jets, m4j):
    combinations = [[[0,1],[2,3]],[[0,2],[1,3]],[[0,3],[1,2]]]
    low_dBB = 9999
    # find dijet combo w/ lowest dBB (difference in the dijet masses)
    for combo in combinations:
        jet0, jet1, jet2, jet3 = jets[combo[0][0]], jets[combo[0][1]], jets[combo[1][0]], jets[combo[1][1]]
        diJet1 = jet0+jet1
        diJet2 = jet2+jet3
        st1, st2 = jet0.Pt() + jet1.Pt(), jet2.Pt() + jet3.Pt()
        dR1, dR2 = jet0.DeltaR(jet1), jet2.DeltaR(jet3)
        if st2 > st1:
            diJet2, diJet1 = diJet1, diJet2
            dR1, dR2 = dR2, dR1
        # institute MDR requirements
        if not ((360/m4j-.5) < dR1 < max((650/m4j+.5), 1.5)): 
            continue
        if not ((235/m4j) < dR2 < max((650/m4j+.7), 1.5)): 
            continue
        dBB = abs(diJet1.M()-diJet2.M())
        if dBB < low_dBB:
            low_dBB = dBB
            low_dijet1 = diJet1
            low_dijet2 = diJet2
    # return the corresponding mHH
    if low_dBB == 9999:
        return 0
    j1 = low_dijet1*(125/(low_dijet1.M()))
    j2 = low_dijet2*(125/(low_dijet2.M()))
    return (j1+j2).M()
    

def make_trees(outpath, file_path, bjets, jet_max):
    if bjets == 3: outpath = outpath[:-1]+'j'
    tree = ToyTree(outpath)
    # loop through and load each run
    for letter in range(7):
        print(chr(letter+ord('B')))
        pico_file = ROOT.TFile(file_path+chr(letter+ord('B'))+'/PicoAOD.root', "READ")
        pico = pico_file.Get("Events")
        for i in range(pico.GetEntries()):
            if i%(pico.GetEntries()//10)==0: print('Event '+str(i)+'/'+str(pico.GetEntries()))
            pico.GetEntry(i)
            # select based on btags and number of jets after pt cut
            if (bjets==3 and pico.fourTag) or (bjets==4 and pico.threeTag): continue
            jets = []
            for j in range(pico.nJet):
                if pico.Jet_pt[j] > 40 and abs(pico.Jet_eta[j]) < 2.5 : jets.append(j)
            if len(jets) < 4 or len(jets) > jet_max: continue
            # need to calculate energy since picos have mass instead
            j0 = ROOT.TLorentzVector()
            j0.SetPtEtaPhiM(pico.canJet0_pt, pico.canJet0_eta, pico.canJet0_phi, pico.canJet0_m)
            j1 = ROOT.TLorentzVector()
            j1.SetPtEtaPhiM(pico.canJet1_pt, pico.canJet1_eta, pico.canJet1_phi, pico.canJet1_m)
            j2 = ROOT.TLorentzVector()
            j2.SetPtEtaPhiM(pico.canJet2_pt, pico.canJet2_eta, pico.canJet2_phi, pico.canJet2_m)
            j3 = ROOT.TLorentzVector()
            j3.SetPtEtaPhiM(pico.canJet3_pt, pico.canJet3_eta, pico.canJet3_phi, pico.canJet3_m)
            pico_mHH = get_mHH([j0,j1,j2,j3], pico.m4j)
            if pico_mHH == 0: continue
            # construct event_data array
            event_data = [len(jets)]
            event_data.append([pico.canJet0_pt, pico.canJet1_pt, pico.canJet2_pt, pico.canJet3_pt])
            event_data.append([pico.canJet0_eta, pico.canJet1_eta, pico.canJet2_eta, pico.canJet3_eta])
            event_data.append([pico.canJet0_phi, pico.canJet1_phi, pico.canJet2_phi, pico.canJet3_phi])
            event_data.append([j0.E(), j1.E(), j2.E(), j3.E()])
            event_data.append(pico.dRjjClose)
            event_data.append(pico.dRjjOther)
            event_data.append(pico.aveAbsEta)
            event_data.append(pico.m4j)
            event_data.append(pico_mHH)
            event_data.append(pico.SB)
            event_data.append(pico.CR)
            event_data.append(pico.SR)
            # fill for each event and write tree at the end
            tree.Fill(event_data)
    tree.Write()
print('Three Tag')
make_trees(outpath, file_path, 3, 100)
print('Four Tag')
make_trees(outpath, file_path, 4, 100)
print('Three Tag')
make_trees(outpath2, file_path, 3, 4)
print('Four Tag')
make_trees(outpath2, file_path, 4, 4)
print('Three Tag')
make_trees(outpath3, file_path, 3, 5)
print('Four Tag')
make_trees(outpath3, file_path, 4, 5)
