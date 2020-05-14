#!/bin/bash

sed "$ d" datacard.txt > datacard_nosys.txt
echo "sys            shape   -               0.000001" >> datacard_nosys.txt