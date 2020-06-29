#!/bin/bash

INPUT=input.csv
OLDIFS=$IFS
IFS=','

[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while read -r XSHIFT YSHIFT SHIFTSCALE PLOTLABEL

do

    sed -i "/shift_scale = .*/c\    shift_scale = $SHIFTSCALE" ./create_data.py
    sed -i "/shift = shift_scale [*] .*/c\    shift = shift_scale * np.array([$XSHIFT, $YSHIFT])" ./create_data.py
    sed -i "/plot_label = .*/c\    plot_label = \"$PLOTLABEL\"" ./create_data.py

    git commit -m "changing data for different plots" create_data.py
    git push origin master

    ssh -t -n gristo@bms3 'cd Simple_Toy_from_Paper && git pull && sh master_plot.sh'

    scp bms3:/home/gristo/Simple_Toy_from_Paper/plots/* /home/risto/Masterarbeit/Simple_Toy_from_Paper/plots/
    scp bms3:/home/gristo/Simple_Toy_from_Paper/*.png /home/risto/Masterarbeit/Simple_Toy_from_Paper/plots/plot_nll_$PLOTLABEL.png

    ssh -t -n gristo@bms3 'cd Simple_Toy_from_Paper && sh cleanup_plots.sh'

    python3 create_data.py

done < $INPUT
IFS=$OLDIFS

#mv ./plots/* ./Plots/

#read -p "Enter Background X-Shift: " XSHIFT
#read -p "Enter Background Y-Shift: " YSHIFT
#read -p "Enter Shift Scale: " SHIFTSCALE
#read -p "Enter Plot Label including CE, SN_no or SD_with: " PLOTLABEL
#
#sed -i "/shift_scale = .*/c\    shift_scale = $SHIFTSCALE" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py
#sed -i "/shift = shift_scale [*] .*/c\    shift = shift_scale * np.array([$XSHIFT, $YSHIFT])" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py
#sed -i "/plot_label = .*/c\    plot_label = '$PLOTLABEL'" /home/risto/Masterarbeit/Simple_Toy_from_Paper/create_data.py
#
#git commit -m "changing data for different plots" create_data.py
#git push origin master
#
#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && git pull && sh master_plot.sh'
#
#scp bms3:/home/gristo/Simple_Toy_from_Paper/plots/* /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/
#scp bms3:/home/gristo/Simple_Toy_from_Paper/*.png /home/risto/Masterarbeit/Simple_Toy_from_Paper/Plots/plot_nll_$PLOTLABEL.png
#
#ssh -t gristo@bms3 'cd Simple_Toy_from_Paper && sh cleanup_plots.sh'
