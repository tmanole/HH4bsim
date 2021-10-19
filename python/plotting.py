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

def setStyle(hist):
    hist.SetLineColor(ROOT.kBlack)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.7)
    hist.SetLineWidth(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.Update()
    

    #hist.GetXaxis().SetLabelFont(43)
    hist.GetXaxis().SetTitleFont(43)
    hist.GetXaxis().SetTitleOffset(3.0)
    hist.GetXaxis().SetLabelSize(1)
    hist.GetXaxis().SetTitleSize(0)
    hist.GetXaxis().SetLabelOffset(0.013)

    hist.GetYaxis().SetLabelSize(18)
    hist.GetYaxis().SetTitleOffset(1.15)
    hist.GetYaxis().SetLabelFont(43)
    hist.GetYaxis().SetTitleFont(43)
    hist.GetYaxis().SetTitleSize(25)

    hist.SetTitleFont(43)


def normHist(hist):
    integral = hist.Integral()
    if integral!=0:
        hist.Scale(1.0/integral)
    else:
        print("normHist integral is zero:",hist.GetName())
    hist.GetYaxis().SetTitle("Normalized Entries")

def plot_production(t4b, t4b_large, sig, t2b, branch, bins, cut, log_scale=False, plot_scalars=False, weights=True, canvas=None, norm=False, branchfit=None, xAxisTitle="", method="", method_name="", ratio_min=0.5, ratio_max=1.5):
    if branchfit is None:
        branchfit = branch

    
#    pi = t4b.GetEntries() * 1.0 / t4b_large.GetEntries()
#    print("Pi Before: ", pi)
#    sum_4 = 0
#    sum_4l = 0
#
#    for i in range(t4b.GetEntries()):
#        t4b.GetEntry(i)
#
#        if t4b.SR == 1:
#            sum_4 += t4b.weight
#
#    for i in range(t4b_large.GetEntries()):
#        t4b_large.GetEntry(i)
#
#        if t4b_large.SR == 1:
#            sum_4l += t4b_large.weight
#
#    pi = sum_4/sum_4l
#    print("After: ", pi)
#    pi = 0.033571298162891484
    pi_4 = 0.0998996846167015

    pi_34 = t4b.GetEntries() * 1.0 / t2b.GetEntries()
    

    t2b.Draw(branchfit+">>h2b_"+branchfit+bins, "w_" + method_name + "*("+cut+")")  #(29038/28017)*
    h2b = ROOT.gDirectory.Get("h2b_"+branchfit)

    t2b.Draw(branchfit+">>hraw_"+branchfit+bins,"weight*("+cut+")")  #(29038/28017)*
    hraw = ROOT.gDirectory.Get("hraw_"+branchfit)

    t4b.Draw(branch+">>h4b_"+branch+bins,"weight*("+cut+")")
    h4b = ROOT.gDirectory.Get("h4b_"+branch)

    t4b_large.Draw(branch+">>h4bL_"+branch+bins, str(pi_4) + "*weight*(" + cut + ")")
    h4bL = ROOT.gDirectory.Get("h4bL_"+branch)

    sig.Draw(branch+">>hsig_"+branch+bins,"weight*("+cut+")")
    hsig = ROOT.gDirectory.Get("hsig_"+branch)

    c=ROOT.TCanvas(branchfit+bins+cut, branch+" "+cut,700,int((500*1.2)))

    rPad = ROOT.TPad("ratio", "ratio", 0, 0.0, 1, 0.3)
    ROOT.gPad.SetTicks(1,1)
    rPad.SetBottomMargin(0.30)
    rPad.SetTopMargin(0.035)
    rPad.SetRightMargin(0.03)
    rPad.SetFillStyle(0)
    rPad.Draw()

    hPad = ROOT.TPad("hist",  "hist",  0, 0.3, 1, 1.0)
    ROOT.gPad.SetTicks(1,1)
    hPad.SetBottomMargin(0.02)
    hPad.SetTopMargin(0.05)
    hPad.SetRightMargin(0.03)
    #hPad.SetLogx()
    
    if log_scale:
        hPad.SetLogy()

    hPad.Draw()

    hPad.SetFillStyle(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gROOT.ForceStyle()

    
    #h2b.SetTitle(branch+" "+cut)
    h2b.SetTitle("")
    h2b.SetFillColor(ROOT.kYellow)
    setStyle(h2b)

    hraw.SetTitle("")
    hraw.SetLineColor(ROOT.kRed)
    hraw.SetLineStyle(1)
    setStyle(hraw)

    setStyle(h4b)
#    h4b.SetLineColor(ROOT.kGray)#ROOT.kBlack)
#    h4b.SetMarkerColor(ROOT.kGray)#ROOT.kBlack)
    #h4b.SetFillColorAlpha(ROOT.kBlack, 0.5)
    h4b.SetMarkerColorAlpha(ROOT.kBlack, 0.5)
    h4b.SetLineColorAlpha(ROOT.kBlack, 0.5)

    setStyle(h4bL)
    h4bL.SetLineColor(ROOT.kBlack)


#    h2b.SetLineColor()

    setStyle(hsig)
    hsig.SetLineColor(ROOT.kBlue)
    hsig.SetLineWidth(3)
    hsig.SetLineStyle(7)        

    h2b.GetYaxis().SetTitle("Entries")
    h4b.GetYaxis().SetTitle("Entries")

    if norm:
        normHist(h2b)
        normHist(h4b)
        normHist(hraw)
        normHist(h4bL) 
        normHist(hsig)


    print("Fit normalization: ", h2b.Integral())
    print("True normalization: ", h4b.Integral())
    print("Fit/True normalizaation: ", h2b.Integral()/h4b.Integral())

    max4b = h4b.GetMaximum()
    max2b = h2b.GetMaximum()
    hMax = max(max2b, max4b)*1.1
    h2b.SetMaximum(hMax)
    h4b.SetMaximum(hMax)

    ratio = ROOT.TH1F(h4bL).Clone()

    ratio.GetXaxis().SetLabelSize(0)
    ratio.SetTitle("")
    ratio.GetYaxis().SetNdivisions(503)
    ratio.SetName(h4bL.GetName().replace("h4b","h4bOver2b"))
    ratio.SetStats(0)

    ratio.Divide(h2b)

    if method == "benchmark":
        ratio.SetYTitle("4b/3b")

    else:
        ratio.SetYTitle("Data/Bkg")

    ratio.SetMinimum(ratio_min)
    ratio.SetMaximum(ratio_max)
    ratio.SetXTitle(xAxisTitle)
    setStyle(ratio)

    #ratio.GetXaxis().SetLabelSize(30)
    ratio.GetXaxis().SetLabelSize(0.1)
    ratio.GetXaxis().SetLabelOffset(0)

    ratio.GetYaxis().SetLabelOffset(0.015)
    ratio.GetYaxis().SetTitleOffset(1.1 )
    ratio.GetXaxis().SetTitleSize(25)



    hPad.cd()
    h2b.Draw("HIST")
    h4b.Draw("ex0 PE SAME")

    h4bL.Draw("ex0 PE SAME")
    hsig.Draw("HIST SAME")

    rPad.cd()
    ratio.Draw("x0 P P0 SAME")

    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.5)

    one = ROOT.TF1("one","1",ratio.GetXaxis().GetXmin(),ratio.GetXaxis().GetXmax())
    one.SetLineColor(ROOT.kBlack)
    one.SetLineWidth(1)
    one.DrawCopy("same l")

    hPad.Update()

    
    hPad.cd()
    l=ROOT.TLegend(0.7,0.7,0.9,0.9)
    l.SetBorderSize(0)
    l.SetFillColor(0)
    l.SetFillStyle(0)
    l.SetTextAlign(12)
    l.AddEntry(h4b, "4b Data", "lp")
    l.AddEntry(h4bL, "4b (x10) Data", "lp")
    l.AddEntry(hsig, "SM HH", "l")

    if method != "benchmark":
        l.AddEntry(h2b, "Bkg Model", "f")

    else:
        l.AddEntry(h2b, "3b Data", "f")

    l.Draw()
    
    region = cut.replace("==1","")
    region = "Signal"    if region == "SR" else region
    region = "Control"   if region == "CR" else region
    region = "Side-band" if region == "SB" else region

    titleRegion  = ROOT.TLatex(0.75, 0.96, "#bf{"+region+" Region}")
    titleRegion.SetTextAlign(11)
    titleRegion.SetNDC()
    titleRegion.Draw()

    if method != "benchmark":
        print(method)
        if method == "resnet":
            method_name = "HH-FvT"

        elif method[:10] == "horizontal":
            if len(method) > 10:
                method_name = "HH-OT, K = " + str(method[12:])

            else:
                method_name = "HH-OT"

        elif method == "resnet_transport":
            method_name = "HH-Comb"

        else:
            method_name = method
    
        titleMethod  = ROOT.TLatex(0.1, 0.96, "#bf{Background Method: "+method_name+"}")

        titleMethod.SetTextAlign(11)
        titleMethod.SetNDC()
        titleMethod.Draw()


        hPad.Update()

        return c, ratio,l, [titleRegion, titleMethod]

    hPad.Update()

    return c, ratio,l, titleRegion

def tree_to_hist(t, branch, bins, weight, cut, hname, hist_file):
    region = cut.replace("==1","")

    pi = 1
   
    scale_fit  = 1#0.33 #* 0.0998996846167015
    scale_3b = 1 #t4b.GetEntries('SR==1')/ t2b1.GetEntries('SR==1')  # * 0.0998996846167015
    #scale_3b = t4b.GetEntries('CR==1') / t2b1.GetEntries('CR==1')
    scale_4b  = 1#0.0998996846167015
    sig_fraction = 3.8/7134 # signal fraction from table 8.1 2016 data https://cds.cern.ch/record/2644551?ln=en

    if not hist_file.GetDirectory(region):
        hist_file.mkdir(region)
    hist_file.cd(region)
    tdir = ROOT.gDirectory
 
    if type(bins) is list: # variable binning
        bins = array('f', bins)
        h  = ROOT.TH1F(hname + '_' +branch, '', len(bins)-1, bins)
        bins = ''
        h.SetDirectory(tdir)

    t.Draw(branch+">>" + hname + "_"  +branch+bins, str(scale_3b)  + "*" + weight + "*("+cut+")")

    h   = ROOT.gDirectory.Get(hname + "_"  +branch)

#####    try:
#####        hsig1.Scale(sig_fraction * h4b.Integral()/hsig1.Integral()) # this method only works for normalizing the signal in the SR where every event is included in the correspoding hist exactly once!!! 
#####        print("Normalize signal")
#####    except:
#####        print("Error!")
#####        pass

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


def plot_fit_hists(hist_file, method_name, plot_vars, x_titles, norm=False):
    pi_4 = 0.0998996846167015

    class standardPlot:
        def __init__(self, region, var):

            if method_name is not None:
            
                self.samples=collections.OrderedDict()
                self.samples[hist_file] = collections.OrderedDict()
    
                self.samples[hist_file]['%s/h4b_large_%s'%(region.name, var.name)] = {
                    "label" : '10#times Statistics True Background',
                    "weight": pi_4,
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "numer A",
                    "color" : "ROOT.kBlack",
                }
        
                self.samples[hist_file]['%s/h4b_%s'%(region.name, var.name)] = {
                    "label" : 'True Background',
                    "legend": 2,
    #                "isData" : True,
    #                "ratio" : "numer A",
                    "color" : "ROOT.kBlack",
                    "alpha" : 0.5,
                    "marker": "1",
                    "lineColor" : "ROOT.kWhite",
                    "lineAlpha": 0,  
                    #"fillColor" : "ROOT.kBlack",
                    #"fillAlpha": 0.5,  
                    "weight": 1,
                    "ratio": "denom A"
                    }
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, method_name, var.name)] = {
                    "label" : "FvT Model",
                    "weight": 1,
                    "legend": 3,
                    "ratio" : "denom A",
                    "stack" : 1,
                    "color" : "ROOT.kYellow"}
        
                self.samples[hist_file]['%s/h_sig_%s'%(region.name, var.name)] = {
                    "label"    : 'SM HH #times100',
                    "legend"   : 6,
                    "weight" : 0.001,
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
                    "legend": 1,
                    "isData" : True,
                    "ratio" : "numer A",
                    "color" : "ROOT.kBlack"}
                self.samples[hist_file]['%s/h_benchmark_%s'%(region.name, var.name)] = {
                    "label" : "Scaled Three-tag Model",
                    "weight": 1,
                    "legend": 2,
                    "ratio" : "denom A",
                    "color" : "ROOT.kOrange+1"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_fvt, var.name)] = {
                    "label" : "FvT Model",
                    "weight": 1,
                    "legend": 3,
                    "ratio" : "denom A",
                    "color" : "ROOT.kRed"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_ot, var.name)] = {
                    "label" : "OT Model",
                    "weight": 1,
                    "legend": 4,
                    "ratio" : "denom A",
                    "color" : "ROOT.kBlue"}
                self.samples[hist_file]['%s/h_%s_%s'%(region.name, hh_comb, var.name)] = {
                    "label" : "OT+FvT Model",
                    "weight": 1,
                    "legend": 5,
                    "ratio" : "denom A",
                    "color" : "ROOT.kViolet-1"}
        
                self.samples[hist_file]['%s/h_sig_%s'%(region.name, var.name)] = {
                    "label"    : 'SM HH #times100',
                    "legend"   : 6,
                    "weight" : 0.001,
                    "color"    : "ROOT.kGreen+3"}
        
                if norm:
                    for (_,s) in self.samples[hist_file].items():
                        s.update({"normalize" : 1}) 
    
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
            variables.append(variable(plot_vars[i], x_titles[i], divideByBinWidth=100))

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
