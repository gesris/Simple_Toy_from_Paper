#!/bin/bash

source ./setup_cmssw.sh

# Convert shapes and write datacard
python write_shapes.py
python write_datacard.py
#python toy7/check_shapes.py

# Perform scan
combineTool.py -M MultiDimFit -d datacard.txt -m 120 \
        --algo grid \
        -P r \
        --points 101 \
        --setParameterRanges r=0.3,1.7 \
        -n PlotNLL

make_datacard_nosys.sh
combineTool.py -M MultiDimFit -d datacard_nosys.txt -m 120 \
        --algo grid \
        -P r \
        --points 101 \
        --setParameterRanges r=0.3,1.7 \
        -n PlotNLLFreezeSys

# Plot 2*deltaNLL vs POI
# NOTE: We reuse code from the Higgs example here!
python convert_scans.py
bash plot_nll.sh

# Signal strength
combine -M MultiDimFit -d datacard.txt --algo singles --robustFit 1

# Pulls
combine -M FitDiagnostics -d datacard.txt --robustFit 1 -v 1 #--saveShapes
python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py -a fitDiagnostics.root | tee pulls.txt
exit

# Prefit/postfit plots
combineTool.py -M T2W -m 120 -o workspace.root -i datacard.txt
#PostFitShapesFromWorkspace -m 120 -w workspace.root -d datacard.txt -o prefit.root
PostFitShapesFromWorkspace -m 120 -w workspace.root -d datacard.txt -f fitDiagnostics.root:fit_s -o postfit.root --postfit --freeze r
python plot_prefit_postfit.py

# Impacts
combineTool.py -M Impacts -m 120 -d workspace.root --doInitialFit --robustFit 1
combineTool.py -M Impacts -m 120 -d workspace.root --doFits --robustFit 1
combineTool.py -M Impacts -m 120 -d workspace.root --output impacts.json
plotImpacts.py -i impacts.json -o impacts