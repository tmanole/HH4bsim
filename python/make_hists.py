import ROOT
import plotting 
import numpy as np
import sys
import pathlib
import os.path
from array import array

sys.path.insert(0, "transport_scripts")
#import transport_func4b as f4b
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
                "\Delta R_{jj} (Close di-jet pair)", 
                "\Delta R_{jj} (Other di-jet pair)", 
                "< |\eta| >",
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


    bins = [#[200, 226, 256, 291, 331, 377, 431, 493, 565, 649, 747, 861, 1000],
            #[200, 226, 256, 291, 331, 377, 431, 493, 565, 649, 747, 861, 1000],
            #'( 80,200,1000)', #make bins (1000-200)/80=10 GeV wide
            [200, 210, 220, 231, 242, 254, 266, 279, 292, 306, 321, 337, 353, 370, 388, 407, 427, 448, 470, 493, 517, 542, 569, 597, 626, 657, 689, 723, 759, 796, 835, 876, 919, 1000],
            [200, 210, 220, 231, 242, 254, 266, 279, 292, 306, 321, 337, 353, 370, 388, 407, 427, 448, 470, 493, 517, 542, 569, 597, 626, 657, 689, 723, 759, 796, 835, 876, 919, 1000],
#            [200, 210, 220, 231, 242, 254, 266, 279, 292, 306, 321, 337, 353, 370, 388, 407, 427, 448, 470, 493, 517, 542, 569, 597, 626, 657, 689, 723, 759, 796, 835, 876, 919, 1000],
            #'(100,200,1000)',
            #'(100,200,1000)',
            '(25,0,2.5)',
            '(40,0,4)',
            '(25,0,2.5)',
            '(32,40,200)',
            '(32,40,200)',
            '(32,40,200)',
            '(32,40,200)',
            '(100,0,800)',
            '(100,0,800)',
            '(100,0,800)',
            '(100,0,800)',
            '(40,150,550)',
            '(30, 0, 1)',
            '(30, 0, 1)',
        #'(50,0,5)',
           ]

    return variables, out_names, x_titles, bins


def make_base_hist(tree, weight, hist_file, hname, scale=1):#, tree_true_large, tree_sig_HH4b, tree_fit, hist_file=None):

    plot_tree_path = get_plot_tree_path("MG3")

    variables, out_names, x_titles, bins = get_plotting_vars()

    print(variables)

    for region in ["CR","SR"]:
        for i in range(len(variables)):
            print(region, variables[i], x_titles[i], bins[i])
            plotting.tree_to_hist(tree, variables[i], bins[i], weight, region + "==1", hname, hist_file, scale=scale)

def base_hist_setup(tree_true, tree_true_large, tree_sig, hist_file):

#    hist_file = '../results/MG3/summary/hists.root'
#    hist_file = ROOT.TFile(hist_file, 'RECREATE')

    make_base_hist(tree_true, "weight", hist_file, "h4b", scale=1)
    make_base_hist(tree_true_large, "weight", hist_file, "h4b_large", scale=0.0998996846167015)
    make_base_hist(tree_sig, "weight", hist_file, "h_sig", scale=100)
#    make_base_hist(tree_fit, hist_file)

    hist_file.Write()
    hist_file.Close()

def make_fit_hists(tree_fit, hist_file, method, method_name):
#    make_base_hist(tree_fit, "w_" + method_name, hist_file, "h_"+method_name)
    plot_vars, _, x_titles, _ = get_plotting_vars()
    plotting.plot_fit_hists(hist_file, method=method, method_name=method_name, plot_vars=plot_vars, x_titles=x_titles, norm=False) 
    plotting.plot_fit_hists(hist_file, method=method, method_name=method_name, plot_vars=plot_vars, x_titles=x_titles, norm=True) 

def make_summary_hists(tree_fit, hist_file):
#    hh_ot   = "HH_OT__pl_emd_p1_R0_4__K_1"
#    hh_comb = "HH_Comb_FvT__pl_emd_p1_R0_4__cl_np799_l0_01_e10"
#    hh_fvt  = "HH_FvT__cl_np799_l0_01_e10"        
#
#    make_base_hist(tree_fit, "w_" + hh_ot, hist_file, "h_"+hh_ot)
#    make_base_hist(tree_fit, "w_" + hh_comb, hist_file, "h_"+hh_comb)
#    make_base_hist(tree_fit, "w_" + hh_fvt, hist_file, "h_"+hh_fvt)
#    make_base_hist(tree_fit, "w_benchmark", hist_file, "h_benchmark")

    plot_vars, _, x_titles, _ = get_plotting_vars()
    plotting.plot_fit_hists(hist_file, method=None, method_name=None, plot_vars=plot_vars, x_titles=x_titles, norm=False) 
    plotting.plot_fit_hists(hist_file, method=None, method_name=None, plot_vars=plot_vars, x_titles=x_titles, norm=True)


def fit_plots(tree_true, tree_true_large, tree_sig_HH4b, tree_fit, SvB=True, mHH=True, reweight=False, data="MG3", method_name="benchmark", method="benchmark", regions=["SR", "CR", "SB"], fromnp=True, signal=False):

    plot_tree_path = get_plot_tree_path(data)

    variables, out_names, x_titles, bins = get_plotting_vars()

    if signal:
        normalizations = ["unnormalized"]

    else:
        normalizations = ["normalized","unnormalized"]

    regions = ["CR", "SR"] #TODO

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

"""
def make_summary_hists(true_tree, bbbj_tree, sig_tree, data, regions=["SR"], mHH=True, SvB=True, hist_file=None):
#
#            pathlib.Path(sum_path).mkdir(parents=True, exist_ok=True) 
#    
#            for region in regions:
#                for i in range(len(variables)):
#                    print(region + "==1")
#                    print("SvB Full Plot")
#                    print(variables[i])#&&SvB_SR_4b>0.95     &&SvB_SR_4b<0.2
#    
#                    #c,r,l,o=plotting.plot_summary(tree_true, tree_true_large, tree_sig_HH4b, tree_sig_S270HH4b, tree_sig_S280HH4b, tree_fit, variables[i], bins[i], region + "==1", norm = norm, xAxisTitle=x_titles[i],method=method,fit_vs_fit=True)
#                    
#                    c,r,l,o=plotting.plot_summary(true_tree, bbbj_tree, sig_tree, variables[i], bins[i], "SR==1", log_scale=log_scale, norm=norm, xAxisTitle=x_titles[i], method="HH-FvT vs HH-RF", hist_file="../results/"+data+"/summary/hists.root")
#                    c.Draw()  
#                    c.SaveAs(sum_path + out_names[i] + ".pdf")
#    

    variables, out_names, x_titles, bins = get_plotting_vars()

    regions=["CR","SR"]

    for norm in [False,True]:

        for log_scale in [False, True]:

            save_root_hists = not norm and not log_scale # really inefficient to remake histograms from ttree draw just to make log and normalized versions...

            for region in regions:

                sum_path = "../results/" + data + "/summary/" + region + '_' + ("log_" if log_scale else "") + ("normalized" if norm else "unnormalized") + "/"
                pathlib.Path(sum_path).mkdir(parents=True, exist_ok=True)

                cut = '%s==1'%region
                print(cut)

                for i, variable in enumerate(variables):
                    print(i,variable)#&&SvB_SR_4b>0.95     &&SvB_SR_4b<0.2


                    c,r,l,o=plotting.plot_summary(true_tree, bbbj_tree, sig_tree, variable, bins[i], cut=cut,
                                                   log_scale=log_scale, norm=norm, xAxisTitle=x_titles[i], method="HH-FvT vs HH-Comb", hist_file=hist_file, save_root_hists=save_root_hists)
                    c.Draw()
                    c.SaveAs(sum_path + out_names[i] + ".pdf")



"""
