import sys
import time
import collections
sys.path.insert(0, '../../PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools
from ROOT import TFile

hist_file = '../results/MG3/summary/hists.root'

class nameTitle:
    def __init__(self, name, title):
        self.name  = name
        self.title = title

class variable:
    def __init__(self, name, xTitle, yTitle = None, zTitle = None, rebin = None, divideByBinWidth = False, normalize = None, normalizeStack = None, mu_qcd=1, xMin=None):
        self.name   = name
        self.xTitle = xTitle
        self.yTitle = yTitle
        self.zTitle = zTitle
        self.xMin = xMin
        self.rebin  = rebin
        self.divideByBinWidth = divideByBinWidth
        self.normalize = normalize
        # self.normalizeStack = normalizeStack
        # self.mu_qcd = mu_qcd

class standardPlot:
    def __init__(self, region, var):
        self.samples=collections.OrderedDict()
        self.samples[hist_file] = collections.OrderedDict()

        self.samples[hist_file]['%s/h4b_%s'%(region.name, var.name)] = {
#            "normalize":1,
            "label" : '10#times Statistics True Background',
            "legend": 1,
            "isData" : True,
            "ratio" : "numer A",
            "color" : "ROOT.kBlack"}
        self.samples[hist_file]['%s/h2b_%s'%(region.name, var.name)] = {
#            "normalize":1,
            "label" : "Scaled Three-tag Model",
            "weight": 1,
            "legend": 2,
            "ratio" : "denom A",
            "color" : "ROOT.kOrange+1"}
        self.samples[hist_file]['%s/h2b1_%s'%(region.name, var.name)] = {
#            "normalize":1,
            "label" : "FvT Model",
            "weight": 1,
            "legend": 3,
            "ratio" : "denom A",
            "color" : "ROOT.kRed"}
        self.samples[hist_file]['%s/h2b3_%s'%(region.name, var.name)] = {
#            "normalize":1,
            "label" : "OT Model",
            "weight": 1,
            "legend": 4,
            "ratio" : "denom A",
            "color" : "ROOT.kBlue"}
        self.samples[hist_file]['%s/h2b2_%s'%(region.name, var.name)] = {
#            "normalize":1,
            "label" : "OT+FvT Model",
            "weight": 1,
            "legend": 5,
            "ratio" : "denom A",
            "color" : "ROOT.kViolet-1"}

        self.samples[hist_file]['%s/hsig1_%s'%(region.name, var.name)] = {
#            "normalize":0.05,
            "label"    : 'SM HH #times100',
            "legend"   : 6,
            "weight" : 0.001,
            "color"    : "ROOT.kGreen+3"}
        
        rMin = 0.45
        rMax = 1.55       

        self.parameters = {"titleLeft"   : '#bf{Simulation}',
                           "titleCenter" : region.title,
                           "titleRight"  : 'Models vs. Truth',
                           #"stackErrors" : True,
                           "maxDigits"   : 4,
                           "ratio"     : True,
                           "rMin"      : rMin,
                           "rMax"      : rMax,
                           'xleg'      : [0.6, 0.935],
                           "rTitle"    : "True / Model",
                           "xTitle"    : var.xTitle,
                           "yTitle"    : "Events" if not var.yTitle else var.yTitle,
                           "outputDir" : '../results/MG3/pretty_plots/%s/'%(region.name),
                           "outputName": var.name,
                           }
        if var.divideByBinWidth: self.parameters["divideByBinWidth"] = var.divideByBinWidth
        if var.xMin is not None: self.parameters['xMin'] = var.xMin
        if var.rebin: self.parameters["rebin"] = var.rebin
        #if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
        #if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.3, 0.3+0.33]

    def plot(self, debug=False):
        PlotTools.plot(self.samples, self.parameters, debug)


regions = [nameTitle('CR', 'Control Region'),
           nameTitle('SR', 'Signal Region'),
       ]

variables = [variable('m4j', 'm_{4j} [GeV]', divideByBinWidth=100),
             variable('mHH', 'm_{HH} [GeV]', divideByBinWidth=100),
             variable('SvB', 'SvB Classifier Output'),#, rebin=10),
             variable('FvT', 'FvT Classifier Output'),#, rebin=10),
             variable("dRjjClose", 'Minimum #DeltaR(j,j)'),
             variable("dRjjOther", 'Complement of Minimum #DeltaR(j,j)'),
             variable("aveAbsEta", '<|#eta|>'),
             variable("jetPt[0]", 'Jet_{0} p_{T} [GeV]'),
             variable("jetPt[1]", 'Jet_{1} p_{T} [GeV]'),
             variable("jetPt[2]", 'Jet_{2} p_{T} [GeV]'),
             variable("jetPt[3]", 'Jet_{3} p_{T} [GeV]'),
             variable("jetPt[0]+jetPt[1]+jetPt[2]+jetPt[3]", "Scalar sum of jet p_{T}'s [GeV]"),
             #variable('reweight', 'FvT weight'),
         ]

plots = []
for region in regions:
    for var in variables:
        plots.append(standardPlot(region, var))
for plot in plots: 
    plot.plot()
    plot.parameters['logY'] = True
    #plot.parameters['yMin'] = 10
    plot.plot()
