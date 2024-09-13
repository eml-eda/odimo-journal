#!/bin/bash

ch=$1
strategy=$2

# source run.sh 0.0e+0 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-10 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-9 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-8 mbv1_search_${ch} half ${strategy} now
source run.sh 5.0e-7 mbv1_search_${ch} half ${strategy} 2023-06-17-08:15:23
source run.sh 1.0e-6 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-6 mbv1_search_${ch} half ${strategy} now
# source run.sh 1.0e-5 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-5 mbv1_search_${ch} half ${strategy} now
# source run.sh 7.0e-5 mbv1_search_${ch} half ${strategy} now
# source run.sh 8.0e-5 mbv1_search_${ch} half ${strategy} now
# source run.sh 1.0e-4 mbv1_search_${ch} half ${strategy} now
# source run.sh 5.0e-4 mbv1_search_${ch} half ${strategy} now