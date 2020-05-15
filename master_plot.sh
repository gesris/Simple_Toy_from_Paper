#!/bin/bash

source venv/bin/activate

sh make_plots.sh

echo "\nInitiating mu-fit.\n"
sleep 1s

deactivate

source setup_cmssw.sh

sh fit.sh
    
