#!/bin/bash

source /home/gristo/Simple_Toy_from_Paper/venv/bin/activate

sh make_plots.sh

echo "\nInitiating mu-fit.\n"
sleep 1s

deactivate

source /home/gristo/Simple_Toy_from_Paper/setup_cmssw.sh

sh fit.sh
    