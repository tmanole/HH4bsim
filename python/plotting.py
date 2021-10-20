import ROOT
from array import array
import numpy as np
from copy import copy


import sys
import time
import collections
sys.path.insert(0, '../../PlotTools/python/') #https://github.com/patrickbryant/PlotTools
import PlotTools


ROOT.TGaxis.SetMaxDigits(4)

ROOT.TGaxis.SetExponentOffset(0.035, -0.078, "y")
ROOT.TGaxis.SetExponentOffset(-0.085,0.035,'x')
ROOT.gStyle.SetOptStat(0)

def tree_to_hist(t, branch, bins, weight, cut, hname, hist_file, scale=1):
    region = cut.replace("==1","")
#
#    pi = 1
#   
#    scale_fit  = 1#0.33 #* 0.0998996846167015
#    scale_3b = 1 #t4b.GetEntries('SR==1')/ t2b1.GetEntries('SR==1')  # * 0.0998996846167015
#    #scale_3b = t4b.GetEntries('CR==1') / t2b1.GetEntries('CR==1')
#    scale_4b  = 1#0.0998996846167015
#    sig_fraction = 3.8/7134 # signal fraction from table 8.1 2016 data https://cds.cern.ch/record/2644551?ln=en

    if not hist_file.GetDirectory(region):
        hist_file.mkdir(region)
    hist_file.cd(region)
    tdir = ROOT.gDirectory
 
    if type(bins) is list: # variable binning
        bins = array('f', bins)
        h  = ROOT.TH1F(hname + '_' +branch, '', len(bins)-1, bins)
        bins = ''
        h.SetDirectory(tdir)

    t.Draw(branch+">>" + hname + "_"  +branch+bins, str(scale) + "*" + str(weight) + "*("+cut+")")

    h   = ROOT.gDirectory.Get(hname + "_"  +branch)

    hist_file.Append(h)

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


def plot_fit_hists(hist_file, method, method_name, plot_vars, x_titles, norm=False):
    pi_4 = 0.0998996846167015

    class standardPlot:
        def __init__(self, region, var):

            if method_name is not None:
            
                self.samples=collections.OrderedDict()
                self.samples[hist_file] = collections.OrderedDict()
    
                self.samples[hist_file]['%s/h4b_large_%s'%(region.name, var.name)] = {
                    "label" : 'True Background',
                    "weight": 1,
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "numer A",
                    "color" : "ROOT.kBlack",
                    }
# TODO
#                self.samples[hist_file]['%s/h4b_%s'%(region.name, var.name)] = {
#                    "label" : 'True Background',
#                    "legend": 2,
#                    "color" : "ROOT.kBlack",
#                    #"alpha" : 0.5,
#                    "marker": "1",
#                    #"fillColor" : "ROOT.kBlack",
#                   # "fillAlpha": 0.5,  
#                    "weight": 1,
#                    "ratio": "denom A",
#                    "drawOptions" : "PE x0 E0E2",
#                    }
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, method_name, var.name)] = {
                    "label" : method + " Model",
                    "weight": 1,
                    "legend": 3,
                    "ratio" : "denom A",
                    "stack" : 1,
                    "color" : "ROOT.kYellow"}
        
                self.samples[hist_file]['%s/h_sig_%s'%(region.name, var.name)] = {
                    "label"    : 'SM HH #times 100',
                    "legend"   : 6,
                    "weight"   : 1,
                    "color"    : "ROOT.kGreen+3"}
        
                if norm:
                    for (_,s) in self.samples[hist_file].items():
                        s["normalize"] = 1
    
                
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
                                   "outputDir" : '../results/MG3/' + method_name 
                                                                   + '/plots/' 
                                                                   + ('normalized' if norm else 'unnormalized') 
                                                                   + "/" + region.name + '/',  
                                   "outputName": var.name,
                                   }
                if var.divideByBinWidth: self.parameters["divideByBinWidth"] = var.divideByBinWidth
                if var.xMin is not None: self.parameters['xMin'] = var.xMin
                if var.rebin: self.parameters["rebin"] = var.rebin
                #if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
                #if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.3, 0.3+0.33]

            else:

                hh_ot   = "HH_OT__pl_emd_p1_R0_4__K_1"
                hh_comb = "HH_Comb_FvT__pl_emd_p1_R0_4__cl_np799_l0_01_e10"
                hh_fvt  = "HH_FvT__cl_np799_l0_01_e10"        
        
                self.samples=collections.OrderedDict()
                self.samples[hist_file] = collections.OrderedDict()
        
                self.samples[hist_file]['%s/h4b_%s'%(region.name, var.name)] = {
                    "label" : '10#times Statistics True Background',
                    "weight": 1,
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "numer A",
                    "color" : "ROOT.kBlack"}
                self.samples[hist_file]['%s/h_benchmark_%s'%(region.name, var.name)] = {
                    "label" : "Scaled Three-tag Background",
                    "weight": 1,
                    "legend": 2,
                    "ratio" : "denom A",
                    "color" : "ROOT.kOrange+1"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_fvt, var.name)] = {
                    "label" : "HH-FvT Model",
                    "weight": 1,
                    "legend": 3,
                    "ratio" : "denom A",
                    "color" : "ROOT.kRed"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_ot, var.name)] = {
                    "label" : "HH-OT Model",
                    "weight": 1,
                    "legend": 4,
                    "ratio" : "denom A",
                    "color" : "ROOT.kBlue"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_comb, var.name)] = {
                    "label" : "HH-Comb Model",
                    "weight": 1,
                    "legend": 5,
                    "ratio" : "denom A",
                    "color" : "ROOT.kViolet-1"}
        
                self.samples[hist_file]['%s/h_sig_%s'%(region.name, var.name)] = {
                    "label"    : 'SM HH #times 100',
                    "weight"   : 1,
                    "legend"   : 6,
                    "color"    : "ROOT.kGreen+3"}
                print(norm) 
                if norm:
                    for (_,s) in self.samples[hist_file].items():
                        s.update({"normalize" : 1}) 
                        print("added norm")
    
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
                                   "outputDir" : '../results/MG3/summary_plots/%s/'%(region.name),
                                   "outputDir" : '../results/MG3/summary_plots/'  
                                                                   + ("normalized" if norm else "unnormalized") 
                                                                   + "/" + region.name + '/',  
                                   "outputName": var.name,
                                   }
                if var.divideByBinWidth: self.parameters["divideByBinWidth"] = var.divideByBinWidth
                if var.xMin is not None: self.parameters['xMin'] = var.xMin
                if var.rebin: self.parameters["rebin"] = var.rebin
                #if var.normalizeStack: self.parameters["normalizeStack"] = var.normalizeStack
                #if 'SvB' in var.name and 'SR' in region.name: self.parameters['xleg'] = [0.3, 0.3+0.33]


                print('../results/MG3/summary_plots/'+ ("normalized" if norm else "unnormalized") + "/" + region.name + '/')
        
        def plot(self, debug=False):
            PlotTools.plot(self.samples, self.parameters, debug)
    
    regions = [nameTitle('CR', 'Control Region'),
               nameTitle('SR', 'Signal Region'),
           ]

    variables = []
    for i in range(len(plot_vars)):
        if plot_vars[i] in ["mHH", "m4j"]:
            variables.append(variable(plot_vars[i], x_titles[i], divideByBinWidth=100))

        else:
            variables.append(variable(plot_vars[i], x_titles[i]))

    plots = []
    for region in regions:
        for var in variables:
            plots.append(standardPlot(region, var))
    for plot in plots: 
        plot.plot()
        plot.parameters['logY'] = True
        #plot.parameters['yMin'] = 10
        plot.plot()



#regions = [nameTitle('CR', 'Control Region'),
#           nameTitle('SR', 'Signal Region'),
#       ]
#
#variables = [variable('m4j', 'm_{4j} [GeV]', divideByBinWidth=100),
#             variable('mHH', 'm_{HH} [GeV]', divideByBinWidth=100),
#             variable('SvB', 'SvB Classifier Output'),#, rebin=10),
#             variable('FvT', 'FvT Classifier Output'),#, rebin=10),
#             variable("dRjjClose", 'Minimum #DeltaR(j,j)'),
#             variable("dRjjOther", 'Complement of Minimum #DeltaR(j,j)'),
#             variable("aveAbsEta", '<|#eta|>'),
#             variable("jetPt[0]", 'Jet_{0} p_{T} [GeV]'),
#             variable("jetPt[1]", 'Jet_{1} p_{T} [GeV]'),
#             variable("jetPt[2]", 'Jet_{2} p_{T} [GeV]'),
#             variable("jetPt[3]", 'Jet_{3} p_{T} [GeV]'),
#             variable("jetPt[0]+jetPt[1]+jetPt[2]+jetPt[3]", "Scalar sum of jet p_{T}'s [GeV]"),
#             #variable('reweight', 'FvT weight'),
#         ]
#
#plots = []
#for region in regions:
#    for var in variables:
#        plots.append(standardPlot(region, var))
#for plot in plots:
#    plot.plot()
#    plot.parameters['logY'] = True
#    #plot.parameters['yMin'] = 10
#    plot.plot()
#
