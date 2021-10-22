import sys
sys.path.insert(0, 'classifier')
import pandas as pd
from make_df import make_df
from model_train import modelParameters

import os.path

def validation_classifier(bbbj_tree, bbbj_df_path, bbbb_df_path, data_name="MG3", method_name='', fromnp=False, region='SR'):

    df_bbbb = pd.read_hdf(bbbb_df_path)
    df_bbbb_reg = df_bbbb[df_bbbb[region]]

    df_bbbj = pd.read_hdf(bbbj_df_path)

    if df_bbbj.shape[0] != bbbj_tree.GetEntries():
        print("bbbj lens do not match")
        print(df_bbbj.shape)
        print(bbbj_tree.GetEntries())
    ws = []
    for j in range(bbbj_tree.GetEntries()):
        bbbj_tree.GetEntry(j)
        ws.append(eval("bbbj_tree.w_" + method_name))

    df_bbbj["weight"] = ws
    df_bbbj_reg = df_bbbj[df_bbbj[region]]

#    df_bbbb["weight"] = np.sum(df_df_bbbb["weight"]) 

    model = modelParameters(df_bbbj_reg, df_bbbb_reg,
                            classifier="FvT", 
                            model_path="../results/" + data_name + "/" + method_name + "/fvt_validation" + "/" + region+"/")
    model.trainSetup()
    model.runEpochs(print_all_epochs=True)
