import ROOT
from array import array
import numpy as np
from copy import copy

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

def plot_summary(t4b, t3b, sig, branch, bins, cut='SR==1', plot_scalars=False, canvas=None, log_scale=False, norm=False, branchfit=None, xAxisTitle="", method="", hist_file=None, save_root_hists=True):
    if branchfit is None:
        branchfit = branch

    region = cut.replace("==1","")

    pi = 1
   
    scale_fit  = 1#0.33 #* 0.0998996846167015
    scale_3b = 1 #t4b.GetEntries('SR==1')/ t2b1.GetEntries('SR==1')  # * 0.0998996846167015
    #scale_3b = t4b.GetEntries('CR==1') / t2b1.GetEntries('CR==1')
    scale_4b  = 1#0.0998996846167015
    sig_fraction = 3.8/7134 # signal fraction from table 8.1 2016 data https://cds.cern.ch/record/2644551?ln=en

    if save_root_hists and hist_file is not None:
        if not hist_file.GetDirectory(region):
            hist_file.mkdir(region)
        hist_file.cd(region)
        tdir = ROOT.gDirectory
 
    if type(bins) is list: # variable binning
        bins = array('f', bins)
        h2b   = ROOT.TH1F('h2b_'  +branchfit, '', len(bins)-1, bins)
        h2b1  = ROOT.TH1F('h2b1_' +branchfit, '', len(bins)-1, bins)
        h2b2  = ROOT.TH1F('h2b2_' +branchfit, '', len(bins)-1, bins)
        h2b3  = ROOT.TH1F('h2b3_' +branchfit, '', len(bins)-1, bins)
        h4b   = ROOT.TH1F('h4b_'  +branch,    '', len(bins)-1, bins)
        hsig1 = ROOT.TH1F('hsig1_'+branch,    '', len(bins)-1, bins)
        bins = ''
        h4b.SetDirectory(tdir)

#    t2b1.Draw(branchfit+">>h2b_"  +branchfit+bins, str(scale_3b)  + "*weight*("+cut+")/reweight")
#    t2b1.Draw(branchfit+">>h2b1_" +branchfit+bins, str(scale_fit) + "*weight*("+cut+")")
#    t2b2.Draw(branchfit+">>h2b2_" +branchfit+bins, str(scale_fit) + "*weight*("+cut+")")
#    t2b3.Draw(branchfit+">>h2b3_" +branchfit+bins, str(scale_fit) + "*weight*("+cut+")")
#    t4b .Draw(branch   +">>h4b_"  +branch   +bins, str(scale_4b)  + "*weight*("+cut+")")
#    sig.Draw(branch   +">>hsig_"+branch   +bins, str(pi)        + "*weight*("+cut+")")

    t3b.Draw(branchfit+">>h2b_"  +branchfit+bins, str(scale_3b)  + "*weight*("+cut+")")
    t3b.Draw(branchfit+">>h2b1_" +branchfit+bins, str(scale_fit) + "*w_HH_FvT__cl_np799_l0_01_e10*("+cut+")")
    t3b.Draw(branchfit+">>h2b2_" +branchfit+bins, str(scale_fit) + "*w_HH_Comb_FvT__pl_emd_p1_R0_4__cl_np799_l0_01_e10*("+cut+")")
    t3b.Draw(branchfit+">>h2b3_" +branchfit+bins, str(scale_fit) + "*w_HH_OT__pl_emd_p1_R0_4__K_1*("+cut+")")
    t4b.Draw(branch   +">>h4b_"  +branch   +bins, str(scale_4b)  + "*weight*("+cut+")")
    sig.Draw(branch   +">>hsig1_"+branch   +bins, str(pi)         + "*weight*("+cut+")")

    h2b   = ROOT.gDirectory.Get("h2b_"  +branchfit)
    h2b1  = ROOT.gDirectory.Get("h2b1_" +branchfit)
    h2b2  = ROOT.gDirectory.Get("h2b2_" +branchfit)
    h2b3  = ROOT.gDirectory.Get("h2b3_" +branchfit)
    h4b   = ROOT.gDirectory.Get("h4b_"  +branch)
    hsig1 = ROOT.gDirectory.Get("hsig1_"+branch)

    try:
        hsig1.Scale(sig_fraction * h4b.Integral()/hsig1.Integral()) # this method only works for normalizing the signal in the SR where every event is included in the correspoding hist exactly once!!! 
        print("Normalize signal")
    except:
        print("Error!")
        pass

    # try:
    #     h2b.Scale(h4b.Integral()/h2b.Integral())
    # except:
    #     pass

    if plot_scalars:
        sig2.Draw(branch+">>hsig2_"+branch+bins,str(pi) + "*weight*("+cut+")")
        hsig2 = ROOT.gDirectory.Get("hsig2_"+branch)
    
        sig3.Draw(branch+">>hsig3_"+branch+bins,str(pi) + "*weight*("+cut+")")
        hsig3 = ROOT.gDirectory.Get("hsig3_"+branch)

    if save_root_hists and hist_file is not None:
        # if not hist_file.GetDirectory(region):
        #     hist_file.mkdir(region)
        # hist_file.cd(region)
        # print(ROOT.gDirectory.GetName())
        # h4b  .SetDirectory(ROOT.gDirectory)
        # h2b  .SetName('%s/%s'%(region, h2b  .GetName()))
        # h2b1 .SetName('%s/%s'%(region, h2b1 .GetName()))
        # h2b2 .SetName('%s/%s'%(region, h2b2 .GetName()))
        # h2b3 .SetName('%s/%s'%(region, h2b3 .GetName()))
        # h4b  .SetName('%s/%s'%(region, h4b  .GetName()))
        # hsig.SetName('%s/%s'%(region, hsig.GetName()))

        hist_file.Append(h2b)
        hist_file.Append(h2b1)
        hist_file.Append(h2b2)
        hist_file.Append(h2b3)
        hist_file.Append(h4b)
        hist_file.Append(hsig1)

    try:
        if not h4b.InheritsFrom("TH1"): return
    except: 
        return

    c=ROOT.TCanvas(branchfit+cut, branch+" "+cut,700,int((500*1.2)))

    rPad = ROOT.TPad("ratio", "ratio", 0, 0, 1, 0.3)
    ROOT.gPad.SetTicks(1,1)
    rPad.SetBottomMargin(0.30)
    rPad.SetTopMargin(0.035)
    rPad.SetRightMargin(0.03)
    rPad.SetFillStyle(0)
    rPad.Draw()

    #r2Pad = ROOT.TPad("ratio2", "ratio2", 0, 0.0, 1, 0.25)
    #ROOT.gPad.SetTicks(1,1)
    #r2Pad.SetBottomMargin(0.30)
    #r2Pad.SetTopMargin(0.035)
    #r2Pad.SetRightMargin(0.03)
    #r2Pad.SetFillStyle(0)
    #r2Pad.Draw()

    hPad = ROOT.TPad("hist",  "hist",  0, 0.3, 1, 1.0)
    ROOT.gPad.SetTicks(1,1)
    hPad.SetBottomMargin(0.02)
    hPad.SetTopMargin(0.05)
    hPad.SetRightMargin(0.03)

    if log_scale:
        hPad.SetLogy()

    hPad.Draw()

    hPad.SetFillStyle(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gROOT.ForceStyle()

    try:
        max_heights = [h4b.GetMaximum()]
    except:
        max_heights = []

    cols=[ROOT.kBlue, ROOT.kGreen, ROOT.kYellow, ROOT.kRed]

    i = 0
    for h2 in [h2b1,h2b2, h2b3, h2b]: #h2b3
        setStyle(h2)

        #h2b.SetTitle(branch+" "+cut)
        h2.SetTitle("")
        h2.SetFillColorAlpha(cols[i],0.2)
        #h2.SetFillColorAlpha(cols[i], 0)#ROOT.kYellow)
        h2.SetLineWidth(3) 
        h2.SetLineColor(cols[i])


        h2.GetYaxis().SetTitle("Entries")

        if norm: 
            normHist(h2)

        max_heights.append(h2.GetMaximum())
        #hMax = max(max2b, max4b)*1.1
        #h2b.SetMaximum(hMax)

        i+=1


    #h4b.SetMaximum(hMax)
    max_height = np.max(max_heights)*1.1

    for h in [h4b, h2b1, h2b2, h2b3]:
        h.SetMaximum(max_height)
        print(h.Integral())
    
    setStyle(hsig1)
    hsig1.SetLineColor(ROOT.kBlue)
    hsig1.SetLineWidth(3)
    hsig1.SetLineStyle(7)

    if plot_scalars:
        setStyle(hsig2)
        hsig2.SetLineColor(ROOT.kRed)
        hsig2.SetLineWidth(3)
        hsig2.SetLineStyle(7)

        setStyle(hsig3)
        hsig3.SetLineColor(12)
        hsig3.SetLineWidth(3)
        hsig3.SetLineStyle(7)

    setStyle(h4b)
    h4b.SetLineColor(ROOT.kBlack)
    
    h4b.GetYaxis().SetTitle("Entries")

    if norm:
        normHist(h4b)

    ratio = ROOT.TH1F(h2b2).Clone()
    ratio.GetXaxis().SetLabelSize(0)
    ratio.SetTitle("")
    ratio.GetYaxis().SetNdivisions(503)
    ratio.SetName(h2b1.GetName().replace("h2b1","h2b1Overh2b2"))
    ratio.SetStats(0)

    ratio.Divide(h4b)  ###

#    if method == "benchmark":
#        ratio.SetYTitle("4b/3b")

#    else:
#        ratio.SetYTitle("/FvT")

    ratio.SetMinimum(0.5)
    ratio.SetMaximum(1.5)
    ratio.SetXTitle(xAxisTitle)
    setStyle(ratio)

    #ratio.GetXaxis().SetLabelSize(30)
    ratio.GetXaxis().SetLabelSize(0.1)
    ratio.GetXaxis().SetLabelOffset(0)

    ratio.GetYaxis().SetLabelOffset(0.015)
    ratio.GetYaxis().SetTitleOffset(1.1 )
    ratio.GetXaxis().SetTitleSize(25)

    ratio.SetMarkerStyle(20)
    ratio.SetMarkerSize(0.5)

    ratio2 = ROOT.TH1F(h2b1).Clone()
    ratio2.GetXaxis().SetLabelSize(0)
    ratio2.SetTitle("")
    ratio2.GetYaxis().SetNdivisions(503)
    ratio2.SetName(h2b1.GetName().replace("h2b1","h2b2Overh2b2"))
    ratio2.SetStats(0)

    ratio2.Divide(h4b)  ###
    ratio2.SetYTitle("RF/FvTyyy")

    ratio2.SetMinimum(-0.5)
    ratio2.SetMaximum(0.5)
    ratio2.SetXTitle(xAxisTitle)
    setStyle(ratio2)

    #ratio.GetXaxis().SetLabelSize(30)
    ratio2.GetXaxis().SetLabelSize(0.1)
    ratio2.GetXaxis().SetLabelOffset(0)

    ratio2.GetYaxis().SetLabelOffset(0.015)
    ratio2.GetYaxis().SetTitleOffset(1.1 )
    ratio2.GetXaxis().SetTitleSize(25)

    ratio2.SetMarkerStyle(20)
    ratio2.SetMarkerSize(0.5)

    hPad.cd()
    h2b1.Draw("HIST")
    h2b.Draw("HIST SAME")
    h2b2.Draw("HIST SAME")
    h2b3.Draw("HIST SAME")
    h4b.Draw("ex0 SAME")

    hsig1.Draw("HIST SAME")

    if plot_scalars:
        hsig2.Draw("HIST SAME")
        hsig3.Draw("HIST SAME")


    #h4b.Draw("ex0 PE SAME")
#    h4b.Draw("ex0 axis SAME")

    #hPad.RedrawAxis()


    rPad.cd()
    ratio.Draw("x0 P P0")
    #ratio2.Draw("x0 P0 SAME")

    #r2Pad.cd()
    #ratio2.Draw("x0 P P0")

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
    l.AddEntry(h4b, "4b (x10) Data", "lp")
    l.AddEntry(h2b, "3b Data", "l")
    l.AddEntry(h2b1, "HH-FvT", "l")
    l.AddEntry(h2b2, "HH-Comb", "l")
    l.AddEntry(h2b3, "HH-OT (K=1)", "l")
    l.AddEntry(hsig1, "SM HH", "l")

    if plot_scalars:
        l.AddEntry(hsig2, "Scalar (270 GeV)", "l")
        l.AddEntry(hsig3, "Scalar (280 GeV)", "l")

    l.Draw()
    
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
    
        titleMethod  = ROOT.TLatex(0.1, 0.96, "")#bf{Background Method: "+method_name+"}")

        titleMethod.SetTextAlign(11)
        titleMethod.SetNDC()
        titleMethod.Draw()


        hPad.Update()

        return c, ratio,l, [titleRegion, titleMethod]

    hPad.Update()

    return c, ratio,l, titleRegion

