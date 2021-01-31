#!/bin/bash

python3 create_data.py

echo " "
echo "Initiating training."
echo " "
python3 train_data.py

echo " "
echo "Plotting histogram and NN function."
echo " "
python3 plot_histogram.py
python3 plot_NN_function.py