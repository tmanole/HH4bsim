import ROOT
import time
import sys
from make_hists import *
import pathlib
import pandas as pd
import os

from threading import Thread

###from validation import validation_classifier

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-d', '--data', default='MG3', type=str, help='Nickname for dataset to use.')
parser.add_argument('-m', '--method', default='', type=str, help='Name of method to use.')
parser.add_argument('-p', '--plot', default=False, type=bool, help='Produce plots?')
parser.add_argument('-v', '--validation', default=False, type=bool, help='Run a classifier to compare fit to SR4b?')
parser.add_argument('-f', '--fit', default=False, type=bool, help='Fit the method?')
parser.add_argument('-sp', '--sumplot', default=False, type=bool, help='Produce summary plots?')
parser.add_argument('-uw', '--updateweights', default=False, type=bool, help='Update SvB and FvT weights?')
parser.add_argument('-cl', '--cl', default="np799_lr0_01_e29", type=str, help='Classifier nickname.')
parser.add_argument('-cp', '--cp', default="", type=str, help='Classifier path.')
parser.add_argument('-pl', '--pl', default="emd_p1_R0_4", type=str, help='Optimal transport coupling nickname.')
parser.add_argument('-dp', '--dp', default="emd_p1_R0_4", type=str, help='Distance matrix nickname.')
#parser.add_argument('-sr', '--startregion', default="CR", type=str, help='Region from which to start (typically CR or SB).')
#parser.add_argument('-tr', '--targetregion', default="SR", type=str, help='Regions where the fit should be made (e.g. SB, CR, CRSR, SBCRSR, etc.)')
parser.add_argument('-muh','--make_univariate_hists', default=False, type=bool, help='Make univariate histograms of 3b and 4b vars.')
parser.add_argument('-K','--K', default=1, type=int, help='Number of nearest neighbors for HH-OT method.')
parser.add_argument('-R','--R', default=0.4, type=float, help='R parameter for EMD distance.')
parser.add_argument('-lr','--lrInit', default=0.01, type=float, help='Initial learning rate of trained FvT classifier.')
parser.add_argument('-tbs','--trbatchsize', default=512, type=int, help='Training batch size for trained FvT classifier.')
parser.add_argument('-np','--numpar', default=6, type=int, help=' "Number of parameters" for trained FvT classifier.')

args   = parser.parse_args()

data   = args.data
method = args.method
#region = args.region
fit    = args.fit
K      = args.K
R      = args.R
#p_ot   = args.pot

lrInit           = args.lrInit
train_batch_size = args.trbatchsize
num_params       = args.numpar

cl_id  = args.cl
pl_id  = args.pl

#source = args.startregion
#target = [args.targetregion[i:i+2] for i in range(0, len(args.targetregion), 2)]

source = "CR"
target = ["SR"]


aux_dir = "../"#"/media/tudor/Seagate Portable Drive/Seagate/LHC/"

### Available Datasets (-d).
#
# toy2d:      Two-dimensional toy dataset used for preliminary studies.
# MG1:        Small dataset (~10k) used for preliminary ROOT studies.
# MG2:        Dataaset of realistic size. 
# MG2_small:  Fraction of MG2 of same size as MG1 for testing purposes.
# MG2_weight: Same as MG2 with non-uniform weights, Nov 2020. 
# MG3:        More realistic large dataset, similar to MG2, used to emulate additional jet activity, Nov 2020.

### Method Terminology and description. 
#
# HH-FvT:      Classifier method, with Patrick's HCR classifier. 
#              Assumes a classifier has already been trained between 3b and 4b. 
#              You can train this classifier by running the file methods/fvt/train.py.
#              You can specify the path of the classifier with the "-cp" switch, 
#              or you can make a nickname for the classifier and add it in the 
#              Classifier Dictionary below, and specify it with 
#              the "-cl" switch. This nickname will be used in the path where the fit is saved. 
#
# HH-RF:       Classifier method, with a random forest.
#              Assumes a classifier has already been trained between 3b and 4b. 
#              Again, you can specify the path of the classifier with the "-cp" switch, 
#              or you can make a nickname for the classifier and add it in the lines below, 
#              and specify it with the "-cl" switch. 
#
# HH-OT:       Horizontal method, which does a nearest neighbor lookup to apply an OT map.
#              Assumes that (1) CR3b-CR4b (resp. SB3b-SB4b) distances have been computed,
#              and (2) that a CR3b-SR3b (resp. SB3b-CR3b) OT coupling has been computed. 
#              The paths to these distances are specified with in the "distance_path" and "coupling_path"
#              variables below. Use the respective "-dp" and "-pl" switches above to specify
#              nicknames for the distances and couplings, as defined below. 
#
# HH-Comb-FvT: Combination method, which applies the classifier in-sample using an OT map
#              from SR3b to CR3b. Requires a coupling path and a classifier path similarly
#              as HH-FvT and HH-OT.
# 
# HH-Comb-RF:  Same thing, using a random forest instead of Patrick's classifier.


### Regions.
# -sr is one of "SR", "CR", "SB", and indicates the region where the fit starts
# (e.g. this is where the classifier is trained, or where the OT coupling starts).
# -tr is one of "SB", "SBCR", "SBCRSR", "CR", "CRSR", "SR", and indicates the target
# regions where the fit is produced (e.g. when you have an OT coupling from CR to SR, 
# this has no choice but to be SR). 


### Examples.
# Fit HH-FvT method for the MG2 dataset and produce plots of the result, 
# using the default classifier trained in the sideband, and produce the fit in SB, CR and SR.
#
#    python background_analysis.py -f True -p True -d MG2 -m HH-FvT -sr SB -tr SBCRSR
#
# Do the same thing but now use the default classifier trained in the control region, 
# and fit the model only in the signal region:
#
#    python background_analysis.py -f True -p True -d MG2 -m HH-FvT -sr CR -tr SR
#
# Do the same thing, but now don't make output plots, and use one of the non-default classifiers
#
#    python background_analysis.py -f True -d MG2 -m HH-FvT -sr CR -tr SR -cl overfit
#
# Now produce plots for the above method without having to refit it:
#
#    python background_analysis.py -p True -d MG2 -m HH-FvT -sr CR -tr SR -cl overfit
# 
# Now train a validation classifier on the above fit, without having to refit the method:
#
#    python background_analysis.py -v True -d MG2 -m HH-FvT -sr CR -tr SR -cl overfit
#




### Classifier Dictionary (for HH-FvT, HH-Comb)

# Use -cp path if specified
if args.cp != "":
    classifier_path = args.classifier_path
    cl_id = "path"

# Else use -cl nickname.
else:
    if data == "MG2":
        if method == "HH-FvT" or method == "HH-Comb-FvT":
            if cl_id == "np799_lr0_01_e29":   
                # num_params=799, lrInit=0.01, epoch=29
                classifier_path = "methods/classifier/classifiers/fvt/CR/FvT_ResNet+multijetAttention_6_6_6_np799_lr0.01_epochs40_stdscale_epoch29_loss0.1662.pkl"
            
            elif cl_id == "overfit":
                #classifier_path = "methods/classifier/fvt_fit/FvT_ResNet+multijetAttention_6_6_6_np799_lr0.01_epochs40_stdscale_epoch16_loss0.1663.pkl"
                #classifier_path = "methods/classifier/fvt_fit_CR/FvT_ResNet+multijetAttention_18_18_18_np6031_lr0.01_epochs40_stdscale_epoch22_loss0.1663.pkl"
                classifier_path = "methods/classifier/fvt_fit_CR/FvT_ResNet+multijetAttention_15_15_15_np4264_lr1_epochs40_stdscale_epoch5_loss0.1670.pkl"
            
            else:
                print("Classifier ID not found, reverting to default.")
                cl_id = "np799_lr0_01_e29"
                classifier_path = "methods/classifier/classifiers/fvt/CR/FvT_ResNet+multijetAttention_6_6_6_np799_lr0.01_epochs40_stdscale_epoch29_loss0.1662.pkl"

        if method == "HH-RF" or method == "HH-Comb-RF":
            if cl_id == "mf_auto_d3":
                # Max_features=auto, max_depth=3
                classifier_path = "methods/classifier/classifiers/rf/CR/rf_out.pkl"

            else:
                print("Classifier ID not found, reverting to default.")
                classifier_path = "methods/classifier/classifiers/rf/CR/rf_out.pkl"

    if data=="MG2_weights":
        pass

    if data=="MG3":

        print(data)

        if method == "HH-RF" or method == "HH-Comb-RF":
            classifier_path = "methods/classifier/classifiers/rf/CR/MG3/rf_out.pkl"
            cl_id = "mf_auto_d3"

        #elif method == "HH-FvT" or method == "HH-Comb-FvT":
        else:
            classifier_path = "methods/classifier/classifiers/fvt/CR/MG3/FvT_ResNet_6_6_6_np799_lr0.01_epochs10_stdscale_epoch10_loss0.6823.pkl" 
            cl_id = "np799_l0_01_e10"


### Transport Coupling Dictionary (for HH-OT, HH-Comb, HH-Cond)
###   R: tuning parameter of the EMD distance.
###   p: exponent of the distance (usually p=1 or 2).

if data == "MG2":
    if pl_id == "emd_p1_R0_4":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/R0_4/coupling_block"
    
    elif pl_id == "emd_p1_R1":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/R1/coupling_block"
    
    elif pl_id == "emd_p1_R2":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/R2/coupling_block"
    
    elif pl_id == "emd_p2_R2":
        coupling_path = "/media/tudor/USB20FD/couplings/p2/R2/coupling_block"
    
    elif pl_id == "sqeucl":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/euclidean/CR3b_SR3b/coupling_block"
        R = 2 * np.pi
    
    else:
        print("The optimal transport coupling " + pl_id + " was not found. Revert to default")
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/CR3b_SR3b/R0_4/coupling_block"

if data == "MG3":
    if pl_id == "conditional_p1_R0_4":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG3/conditional/R0_4/coupling_CRSR_block"

    elif pl_id == "unbalanced_p1_R0_4_tau200_lambda10":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG3/CR3b_SR3b/R0_4/unbalanced_tau200_lambda10/unbalanced"

    elif pl_id == "unbalanced_sT_p1_R0_4_tau200_lambda10":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG3/CR3b_SR3b/R0_4/unbalanced_sT_tau200_lambda10/unbalanced"

    elif pl_id == "unbalanced_p1_R0_4":
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG3/CR3b_SR3b/R0_4/unbalanced/unbalanced"

    else:
        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG3/CR3b_SR3b/R0_4/coupling_block"

# Other MG2 couplings. 
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/oracle_R0_4_p1_clean/coupling"            # This is upsample 4b map, balanced.
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/sTdensity_CRSR_R0_4_p1/coupling_block"    # This is balanced CRSR density ratio coupling.
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/oracle_unbalanced_r0_4_p1/tau200_lambda3/coupling"
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/unbalanced_sTdensity_R0_4_p1_tau200_lambda5/coupling_block"
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/unbalanced_sTdensity_3b4b_R0_4_p1_tau50/coupling_block"
#        coupling_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/couplings/MG2/emd_opt4/CR3b_SR3b/unbalanced_sTdensity_CRSR_R0_4_p1_tau200_lambda5/coupling_block"

### Distance Dictionary (for HH-OT)

if data == "MG2":
    if pl_id == "sqeucl":
        distance_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/distances/MG2/nn_euclidean_CR3b_CR4b/tblock"
    else:
        distance_path = aux_dir + "distances/MG2/emd_opt4/nn_CR3b_CR4b/tblock"

if data == "MG3":
    if pl_id == "sqeucl":
        pass 

    else:
        distance_path = "/media/tudor/Seagate Portable Drive/Seagate/LHC/distances/MG3/tblock"

        I_CR3b_hp="../couplings/MG3/ordering_sT/CR3b_SR3b/I_MG3_CR3b.npy"
        I_SR3b_hp="../couplings/MG3/ordering_sT/CR3b_SR3b/I_MG3_SR3b.npy"
        I_CR3b_vp="../distances/MG3/nn_inds/I_nn_CR3b.npy"
        I_CR4b_vp="../distances/MG3/nn_inds/I_nn_CR4b.npy"

###  Block Index Dictionary (for OT)
if data == "MG2":
    I_source = aux_dir + "couplings/MG2/ordering_sT/CR3b_SR3b/I_CR3b.npy"
    I_target = aux_dir + "couplings/MG2/ordering_sT/CR3b_SR3b/I_SR3b.npy"

if data == "MG3":
    I_source = aux_dir + "couplings/MG3/ordering_sT/CR3b_SR3b/I_MG3_CR3b.npy"
    I_target = aux_dir + "couplings/MG3/ordering_sT/CR3b_SR3b/I_MG3_SR3b.npy"

### Method Naming
#method_name = method

def get_method_name(method):
    if method == "HH-OT":
        return "HH_OT__pl_" + pl_id + "__K_" + str(K)

    if method == "HH-FvT":
        return "HH_FvT__cl_" + cl_id 

    if method == "HH-RF":
        return "HH_RF__cl_" + cl_id 

    if method == "HH-Comb-FvT":
        return "HH_Comb_FvT__pl_" + pl_id + "__cl_" + cl_id 

    if method == "HH-Comb-RF":
        return "HH_Comb_RF__pl_" + pl_id + "__cl_" + cl_id 

    return method

method_name = get_method_name(method)

print(method_name)

out_path = "../results/" + data + "/" + method_name + "/"

pathlib.Path(out_path).mkdir(parents=True, exist_ok=True) 

out_path += "fit.root"

plot_reweight_distr=False
if method in ["HH-FvT", "HH-RF"]:
    plot_reweight_distr = True


### Load event files.
bs = data == "MG1" #args.bquarks
bid = "bbjj" if bs == 2 else "bbbj"
bb  = str(bs) + "b"

print("Loading TTrees.")
if args.plot or args.fit or args.updateweights:
    ### ROOT TFiles
    bbbb_file = ROOT.TFile(aux_dir + "events/" + data + "/TTree/bbbb.root", "READ")
    bbbj_file = ROOT.TFile(aux_dir + "events/" + data + "/TTree/" + bid + ".root", "READ")
    bbbb_large_file = ROOT.TFile(aux_dir + "events/" + data + "/TTree/bbbb_large.root", "READ")
    
    sig_HH4b_file = ROOT.TFile(aux_dir + "events/" + data + "/TTree/HH4b_dR04_toyTree.root", "READ")
    
    ### ROOT TTrees
    bbbb_tree = bbbb_file.Get("Tree")
    bbbj_tree = bbbj_file.Get("Tree")
    bbbb_large_tree = bbbb_large_file.Get("Tree")
    
    sig_HH4b_tree = sig_HH4b_file.Get("Tree")
    
    df3b = pd.read_hdf(aux_dir + "events/" + data + "/dataframes/" + bid + ".h5")
    df4b = pd.read_hdf(aux_dir + "events/" + data + "/dataframes/bbbb.h5")

### Make univariate hists if asked.
print("Making univariate histograms.")

if args.make_univariate_hists:
    make_univariate_hists(bbbj_tree, '3b', data=data)
    make_univariate_hists(bbbb_tree, '4b', data=data)
    make_dijet_masses(bbbj_tree, region='SR', four_tag='3b', data=data)
    make_dijet_masses(bbbb_tree, region='SR', four_tag='4b', data=data)


### Update SvB Weights

##### Call fns under SvB dir




### Update FvT Weights

#### Call fns under FvT dir



### Start fit.
if not fit:

    if (args.plot or args.validation) and not os.path.exists(out_path):
            sys.exit("Existing fit for method " + method + " not found. Try using the -f switch to fit the method.")

else: 
    sys.path.insert(0, 'methods')

    if method == "benchmark":
        print("Starting Benchmark.")
        import benchmark

        benchmark.benchmark(bbbj_tree, bbbb_tree, out_path)
 
    elif method == "HH-Comb-FvT":
        print("Starting HH-Comb-FvT fit.")
        import combination

        combination.resnet_large_transport(bbbj_tree, bbbb_tree,out_path, method_name, coupling_path, I_source, I_target, source=source, fvt=True)

    elif method == "HH-Comb-RF":
        print("Starting HH-Comb-FvT fit.")
        import resnet_transport

        resnet_transport.resnet_large_transport(bbbj_tree_SR, df3b, df4b, classifier_path, out_path, coupling_path, I_source, I_target, source=source, fvt=False)

    elif method == "HH-RF":
        print("Starting FvT fit from control region.")
        import evaluate_classifier

        evaluate_classifier.fit(bbbj=bbbj_tree, bbbb=bbbb_tree, df3b=df3b, df4b=df4b, fvt=False, 
                                classifier_path=classifier_path, out_path=out_path, source=source, target=target)
 
    elif method == "HH-FvT":
        print("Starting FvT fit from control region.")
        import evaluate_classifier

        evaluate_classifier.fit(bbbj=bbbj_tree, bbbb=bbbb_tree, method_name=method_name, 
                                       out_path=out_path, source=source, target=target, lrInit=lrInit, 
                                       train_batch_size=train_batch_size, num_params=num_params, fvt=True)
 
    elif method == "HH-OT":
        print("Starting horizontal transport fit.")
        import nn_transport

        nn_transport.nn_large_transport(
            bbbj=bbbj_tree,
            bbbb=bbbb_tree, 
            out_path=out_path,
            method_name=method_name, 
            K=K, R=R, coupling_path=coupling_path, 
            distance_path=distance_path,
            I_CR3b_hp=I_CR3b_hp,
            I_SR3b_hp=I_SR3b_hp,
            I_CR3b_vp=I_CR3b_vp,
            I_CR4b_vp=I_CR4b_vp
        )

    else:
        raise Exception("Method not recognized")

    
    bbbj_file.Close()

    fit_file = ROOT.TFile(out_path, "READ")
    fit_tree = fit_file.Get("Tree")
    
    bbbj_file = ROOT.TFile(aux_dir + "events/" + data + "/TTree/" + bid + ".root", "RECREATE")
    bbbj_tree = fit_tree.CloneTree()
    bbbj_file.Write()
    fit_file.Close()



plotting = args.plot or args.sumplot


#if args.fit or plotting or args.updateweights or args.updateweights:
#    tree_names = []
#    trees = []
#    weight_names = []
#    pl_regions = []
#
#    if args.updateweights:
#        tree_names = ["true", "true_large", "sig_HH4b", "sig_S270HH4b", "sig_S280HH4b"]
#        trees = [bbbb_tree, bbbb_large_tree, sig_HH4b_tree, sig_S270HH4b_tree, sig_S280HH4b_tree]
#        weight_names = ["true", "true_large", "sig_HH4b", "sig_S270HH4b", "sig_S280HH4b"]
#        pl_regions = ["CRSR", "CRSR", "CRSR", "CRSR", "CRSR"]
#
#    if args.plot or args.fit:
#        tree_names.append(method_name)
#        trees.append(fit_tree)
#        weight_names.append("3b")
#        pl_regions.append(args.targetregion)
#
#    prepare_plot_trees(trees, tree_names, weight_names, pl_regions, data=data)

if plotting: # or args.validation:

#    if method == "HH-FvT" or method == "HH-RF":
#        f2 = ROOT.TFile("../results/" + data + "/" + method_name + "/true.root")
#        bbbb_tree = f2.Get("Tree")

    mHH = (data != 'MG1')

    # Put a flag here to make sure user has all requisite files.

#    plot_tree_path = get_plot_tree_path(data)
#
#    file_true =       ROOT.TFile(plot_tree_path + "true.root", "READ")
#    file_true_large = ROOT.TFile(plot_tree_path + "true_large.root", "READ")
#    file_sig1 =       ROOT.TFile(plot_tree_path + "sig_HH4b.root", "READ")
#
#    tree_true_large_weighted = file_true_large.Get("Tree")
#    tree_sig_HH4b_weighted = file_sig1.Get("Tree")

    if args.plot:
#        file_fit =        ROOT.TFile(plot_tree_path + tree_names[-1] + ".root", "READ")
#        tree_fit = file_fit.Get("Tree")

#        if plot_reweight_distr:
#            f_rew = ROOT.TFile("../results/" + data + "/" + method_name+ "/true.root")
#            tree_true_reweight = f_rew.Get("Tree")
#
#        else:
#            tree_true_reweight = False
#
        fit_plots(tree_true=bbbb_tree,
                  tree_true_large=bbbb_large_tree,
                  tree_sig_HH4b=sig_HH4b_tree,
                  tree_fit=bbbj_tree,
                  SvB=True,
                  mHH = mHH,
                  reweight=plot_reweight_distr,
                  data=data,
                  method_name=method_name,
                  method=method,
                  regions=target,
                  fromnp=False)


    # Next step: Merge this step with make_summary_hists below. Use these updated trees therein. 
    #   

"""
    if args.sumplot:
        methods = ["HH-FvT", "HH-Comb-FvT", "HH-OT"]
        method_names = [get_method_name(m) for m in methods]
        out_paths = ["../results/" + data + "/plot_trees/" + m + ".root" for m in method_names]
        print(out_paths)
#        [pathlib.Path(o).mkdir(parents=True, exist_ok=True) for o in out_paths]
#        out_paths = [o + "fit.root" for o in out_paths]

        tfiles = [ROOT.TFile(o, "READ") for o in out_paths]
        ttrees = [f.Get("Tree") for f in tfiles]

    #    if method == "HH-FvT" or method == "HH-RF":
    #        f2 = ROOT.TFile("../results/" + data + "/" + method_name + "/true.root")
    #        bbbb_tree = f2.Get("Tree")

        make_summary_hists(
                    tree_true_weighted,
#                    tree_true_large_weighted, 
                    ttrees[0],
                    ttrees[1],
                    ttrees[2],
                    tree_sig_HH4b_weighted,
                    tree_sig_S270HH4b_weighted,
                    tree_sig_S280HH4b_weighted,
#                   tree_true=tree_true_weighted, 
#                   tree_true,
#                   ttrees[0],            
#                   ttrees[1], 
#                   sig_HH4b_tree, 
#                   sig_S270HH4b_tree, 
#                   sig_S280HH4b_tree,
                   data=data
                  )

if args.validation:
    region = target[-1]
    pathlib.Path(out_path.replace("fit.root", "") +"fvt_validation/" + region).mkdir(parents=True, exist_ok=True)
    validation_classifier(fit_tree, aux_dir + "events/" + data + "/dataframes/bbbb_large.h5", data_name=data, method_name=method_name, region=region, epochs=40, num_params=num_params, lrInit=lrInit, train_batch_size=train_batch_size)

#              SvB=True, 
#              mHH = mHH, 
#              reweight=reweight, 
#              data=data, 
#              method_name=method_name, 
#              method=method, 
#              regions=target, 
#              fromnp=False)
"""
