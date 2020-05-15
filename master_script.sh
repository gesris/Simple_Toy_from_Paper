#!/bin/bash

read -p "Enter Background X-Shift: " XSHIFT
read -p "Enter Background Y-Shift: " YSHIFT
read -p "Enter Shift Scale: " SHIFTSCALE
read -p "Enter Plot Label including CE, SN_no or SD_with: " PLOTLABEL


#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sed -i "/shift_scale = .*/c\    shift_scale = $SHIFTSCALE" ./create_data.py'
#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sed -i "/shift = shift_scale [*] .*/c\    shift = shift_scale * np.array([$XSHIFT, $YSHIFT])" ./create_data.py'
#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sed -i "/plot_label = .*/c\    plot_label = \"$PLOTLABEL\"" ./create_data.py'

#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && cat ./create_data.py'


sed -i "/shift_scale = .*/c\    shift_scale = $SHIFTSCALE" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py
sed -i "/shift = shift_scale [*] .*/c\    shift = shift_scale * np.array([$XSHIFT, $YSHIFT])" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py
sed -i "/plot_label = .*/c\    plot_label = '$PLOTLABEL'" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py

git commit -m "changing data for different plots" create_data.py
git push origin master

ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && git pull && sh master_plot.sh'

scp bms3:/home/gristo/Simple_Toy_from_Paper/plots/* /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/
scp bms3:/home/gristo/Simple_Toy_from_Paper/*.png /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/

ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sh cleanup_plots.sh'