import ROOT
import plotting 
import numpy as np
import sys
import pathlib
import os.path
from array import array

sys.path.insert(0, "transport_scripts")
import transport_func4b as f4b
sys.path.insert(0, ".")
    
def get_plot_tree_path(data):
    path = "../results/" + data + "/plot_trees/"
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path 

def get_plotting_vars():
    variables = ["m4j", 
                 "mHH",
                 "dRjjClose", 
                 "dRjjOther", 
                 "aveAbsEta",
                 "jetPt[0]",
                 "jetPt[1]",
                 "jetPt[2]",
                 "jetPt[3]",
                 "jetEnergy[0]",
                 "jetEnergy[1]",
                 "jetEnergy[2]",
                 "jetEnergy[3]",
                 "jetPt[0]+jetPt[1]+jetPt[2]+jetPt[3]",
                 "SvB",
                 "FvT",                
                ]

    out_names = ["m4j", 
                 "mHH",
                 "dRjjClose", 
                 "dRjjOther", 
                 "aveAbsEta",
                 "jetPt0",
                 "jetPt1",
                 "jetPt2",
                 "jetPt3",
                 "jetEnergy0",
                 "jetEnergy1",
                 "jetEnergy2",
                 "jetEnergy3",
                 "sT",
                 "SvB",
                 "FvT",
                ]


    x_titles = ["Four-jet mass [GeV]", 
                "mHH [GeV]",
                "#Delta R_{jj} (Close di-jet pair)", 
                "#Delta R_{jj} (Other di-jet pair)", 
                "< |#eta| >",
                "jetPt_{0} (Transverse momentum of first jet)",
                "jetPt_{1} (Transverse momentum of second jet)",
                "jetPt_{2} (Transverse momentum of second jet)",
                "jetPt_{3} (Transverse momentum of second jet)",
                "jetEnergy_{0} (Energy of first jet)",
                "jetEnergy_{1} (Energy of second jet)",
                "jetEnergy_{2} (Energy of second jet)",
                "jetEnergy_{3} (Energy of second jet)",
                "s_{T} (Scalar sum of transverse momenta)",
                "Signal vs. Background Classifier Output",
                "Four- vs. Three-Tag Classifier Output",
              ]

    bins = ['(100,200,1000)', 
            '(100,200,1000)',
            '(100,0,2.5)', 
            '(100,0,4)', 
            '(100,0,2.5)',
            '(100,40,200)', 
            '(100,40,200)', 
            '(100,40,200)', 
            '(100,40,200)', 
            '(100,0,800)', 
            '(100,0,800)', 
            '(100,0,800)', 
            '(100,0,800)', 
            '(100,150,550)', 
            '(100, 0, 1)',
            '(100, 0, 1)',
           ]


    return variables, out_names, x_titles, bins


def fit_plots(tree_true, tree_true_large, tree_sig_HH4b, tree_fit, SvB=True, mHH=True, reweight=False, data="MG3", method_name="benchmark", method="benchmark", regions=["SR", "CR", "SB"], fromnp=True, signal=False):

    plot_tree_path = get_plot_tree_path(data)

    variables, out_names, x_titles, bins = get_plotting_vars()

    if signal:
        normalizations = ["unnormalized"]

    else:
        normalizations = ["normalized","unnormalized"]

    for norm in [False,True]:
        for do_log in [False,True]:
            for region in regions:
                for i in range(len(variables)):
                    print(region + "==1")
                    print("SvB Full Plot")
                    print(variables[i])#&&SvB_SR_4b>0.95     &&SvB_SR_4b<0.2

                    c,r,l,o=plotting.plot_production(tree_true, tree_true_large, tree_sig_HH4b,tree_fit, variables[i], bins[i], region + "==1", log_scale=do_log, norm = norm, xAxisTitle=x_titles[i],method=method, method_name=method_name)
                    c.Draw()

                    outpath = "../results/" + data + "/" + method_name + "/plots/" + ("log_" if do_log else "") + ("normalized" if norm else "unnormalized") + "/" + region
#                    outpath = "../results/" + data + "/" + method_name + "/plots/" + ("normalized" if norm else "unnormalized") + "/" + region 
                    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True) 

                    c.SaveAs(outpath + "/" + out_names[i] + ".pdf")

##    if reweight:
##        for normalization in normalizations:
##            for region in regions:
##                c,r,l,o=plotting.plot(tree_true_reweight, tree_fit, "reweight", "(100,0,3)", region + "==1", weights=False, norm = (normalization=="normalized"), xAxisTitle=x_titles[i],method=method, ratio_min=0, ratio_max=2.5, fit_vs_fit=False)
##                c.Draw()
##                c.SaveAs("../results/" + data + "/" + method_name + "/plots/SvB/" + normalization + "/" + region + "/reweight_raw.pdf")

#    variables = ["reweight"]
#    out_names = ["reweight"]
#    x_titles  = ["Reweight"]
#    bins      = ["(100,0,3)"]
#
#    for norm in [False,True]:
#       for region in regions:
#            for i in range(len(variables)):
#                c,r,l,o=plotting.plot(tree_true, tree_fit, variables[i], bins[i], region + "==1", norm = False, xAxisTitle=x_titles[i],method=method,fit_vs_fit=True)
#                c,r,l,o=plotting.plot_production(tree_true, tree_true_large, tree_sig_HH4b, tree_sig_S270HH4b, tree_sig_S280HH4b, tree_fit, variables[i], bins[i], region + "==1", norm = norm, xAxisTitle=x_titles[i],method=method,fit_vs_fit=True)
#
#                c.Draw()
#
#                outpath = "../results/" + data + "/" + method_name + "/plots/SvB/" + ("normalized" if norm else "unnormalized") + "/" + region 
#                pathlib.Path(outpath).mkdir(parents=True, exist_ok=True) 
#
#                c.SaveAs(outpath + "/" + out_names[i] + ".pdf")

def make_univariate_hists(tree, four_tag, mHH=True, data="MG2_small", fromnp=True):
    variables, out_names, x_titles, bins = get_plotting_vars(mHH=mHH, SvB=SvB)

    if mHH:
        variables.append("mHH")
        bins.append("(100,80,800)")

    else:
        variables.append("mZZ")
        bins.append("(100,100,800)")

    for region in ["SB", "CR", "SR"]:
        for i in range(len(variables)):
            tree.Draw(variables[i]+">>h_tree_" + variables[i] + bins[i], region + "==1")
            h_tree = ROOT.gDirectory.Get("h_tree_" + variables[i])
            h_tree.SetLineColor(ROOT.kBlack)
            h_tree.SetFillColor(ROOT.kYellow)

            c=ROOT.TCanvas()
            h_tree.Draw("HIST")
            c.Draw()
            c.SaveAs("../results/eda/" + data + "/" + four_tag  + "/" + region + "/" + variables[i] + ".pdf")


def make_dijet_masses(tree, region, four_tag, data="MG2_small", fromnp=False):
    c, h, ind = f4b.dijet_mass_plane(tree, CR=0, SR=1, SB=0, fromnp=fromnp)
    h.Draw("COLZ")
    c.Draw()
    c.SaveAs("../results/eda/" + data + "/" + four_tag + "/" + region + "/dijet_mass.pdf")


def make_summary_hists(true_tree, comb_tree, fvt_tree, hh_ot_tree, sig1_tree, sig2_tree, sig3_tree, data, regions=["SR"], mHH=True, SvB=True):

    variables, out_names, x_titles, bins = get_plotting_vars()

    for norm in [False,True]:

        for log_scale in [False, True]:

            sum_path = "../results/" + data + "/summary/" + ("log_" if log_scale else "") + ("normalized" if norm else "unnormalized") + "/"
            pathlib.Path(sum_path).mkdir(parents=True, exist_ok=True) 
    
            for region in regions:
                for i in range(len(variables)):
                    print(region + "==1")
                    print("SvB Full Plot")
                    print(variables[i])#&&SvB_SR_4b>0.95     &&SvB_SR_4b<0.2
    
                    #c,r,l,o=plotting.plot_summary(tree_true, tree_true_large, tree_sig_HH4b, tree_sig_S270HH4b, tree_sig_S280HH4b, tree_fit, variables[i], bins[i], region + "==1", norm = norm, xAxisTitle=x_titles[i],method=method,fit_vs_fit=True)
                    
                    c,r,l,o=plotting.plot_summary(true_tree, fvt_tree, comb_tree, hh_ot_tree, sig1_tree, sig2_tree, sig3_tree, variables[i], bins[i], "SR==1", log_scale=log_scale, norm=norm, xAxisTitle=x_titles[i], method="HH-FvT vs HH-RF")
                    c.Draw()  
                    c.SaveAs(sum_path + out_names[i] + ".pdf")
    
