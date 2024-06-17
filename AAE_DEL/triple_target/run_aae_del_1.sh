#!/usr/bin/bash

# Comment out this line if your shell has already been setup
#conda init bash

# If your environment name is different,
# Please Update it here
# You can also remove this if you plan to set your environment manually
#conda activate del_aae

# declare -a datasets=("PCBA" "ZINC")

# for n in 1 2 3; do
#   # shellcheck disable=SC2068
#   for dataset in ${datasets[@]}; do
#     python manage.py del --dataset "${dataset}" --random_seed "$((131+n))" --ranking fnds --batch_size 256 --use_gpu
#     python manage.py del --dataset "${dataset}" --random_seed "$((141+n))" --ranking sopr --batch_size 256 --use_gpu
#   done
# done

declare -a datasets=("ZINC")
for dataset in ${datasets[@]}
do

python manage.py del --dataset "${dataset}" --random_seed "$((141+n))" --ranking fndr --batch_size 128 --use_gpu

done
