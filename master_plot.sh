#!/bin/bash

source /home/gristo/Simple_Toy_from_Paper/venv/bin/activate

cd /home/gristo/Simple_Toy_from_Paper/

python3 create_data.py
python3 train_data.py
python3 plot_histogram.py
python3 plot_NN_function.py


    