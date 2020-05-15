#!/bin/bash

source /home/gristo/Simple_Toy_from_Paper/venv/bin/activate

sh /home/gristo/Simple_Toy_from_Paper/make_plots.sh

echo "Initiating mu-fit."
sleep 1s

deactivate

source /home/gristo/Simple_Toy_from_Paper/setup_cmssw.sh

sh /home/gristo/Simple_Toy_from_Paper/fit.sh
    