#!/bin/bash

ssh -t gristo@deepthought 'cd Simple_Toy_from_Paper && git pull && sh master_plot.sh'
ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sh master_fit.sh'

scp bms3:/home/gristo/Simple_Toy_from_Paper/plots/* /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/
scp bms3:/home/gristo/Simple_Toy_from_Paper/*.png /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/