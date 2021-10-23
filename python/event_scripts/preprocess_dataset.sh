#!/bin/bash

# You should be running this script from toy4b/python/event_scripts. 

DATA=MG3   # Modify this to the nickname of your dataset.


TTREE_PATH=../../events/$DATA/TTree   # This script assumes this path exists already.
DF_PATH=../../events/$DATA/dataframes
PT_PATH=../../events/$DATA/PtEtaPhi

mkdir $DF_PATH
mkdir $PT_PATH

# Split ROOT files by region.
python tree_regions.py -pi $TTREE_PATH/bbbb.root -po $TTREE_PATH/bbbb_
python tree_regions.py -pi $TTREE_PATH/bbbj.root -po $TTREE_PATH/bbbj_

# ROOT --> PtEtaPhi.
python tree_to_PtEtaPhi.py -pi $TTREE_PATH/bbbb.root -po $PT_PATH/ -bs 4
python tree_to_PtEtaPhi.py -pi $TTREE_PATH/bbbj.root -po $PT_PATH/ -bs 3 

# ROOT --> h5.
python tree_to_df.py -pi $TTREE_PATH/bbbb.root -po $DF_PATH/bbbb.h5 -f True
python tree_to_df.py -pi $TTREE_PATH/bbbb_large.root -po $DF_PATH/bbbb_large.h5 -f True
#python tree_to_df.py -pi $TTREE_PATH/bbbb_SB.root -po $DF_PATH/bbbb_SB.h5 -f True
#python tree_to_df.py -pi $TTREE_PATH/bbbb_CR.root -po $DF_PATH/bbbb_CR.h5 -f True
#python tree_to_df.py -pi $TTREE_PATH/bbbb_SR.root -po $DF_PATH/bbbb_SR.h5 -f True

python tree_to_df.py -pi $TTREE_PATH/bbbj.root -po $DF_PATH/bbbj.h5 
#python tree_to_df.py -pi $TTREE_PATH/bbbj_SB.root -po $DF_PATH/bbbj_SB.h5
#python tree_to_df.py -pi $TTREE_PATH/bbbj_CR.root -po $DF_PATH/bbbj_CR.h5
#python tree_to_df.py -pi $TTREE_PATH/bbbj_SR.root -po $DF_PATH/bbbj_SR.h5


