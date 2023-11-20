#! /bin/bash


path=./

mkdir -p Output/


nvidia-smi
echo -n "Enter gpu device number >"
read gpu_id
echo "gpu device number is : $gpu_id "
if [ $gpu_id = 0 ]
then
    export CUDA_VISIBLE_DEVICES=GPU-4ec871ec-32f1-ed5d-468e-3991153506f0 # V100, device #0
    echo Device $dev_num, UUID: $CUDA_VISIBLE_DEVICES
elif [ $gpu_id = 1 ]
then
    export CUDA_VISIBLE_DEVICES=GPU-279dc233-a0f5-179b-c9c3-f493884df5b5 # V100, device #1
    echo Device $dev_num, UUID: $CUDA_VISIBLE_DEVICES
elif [ $gpu_id = 2 ]
then
    export CUDA_VISIBLE_DEVICES=GPU-bdccd9ce-cd29-c0e5-397d-139e6808376b # A100, device #2
    echo Device $dev_num, UUID: $CUDA_VISIBLE_DEVICES
elif [ $gpu_id = 3 ]
then
    export CUDA_VISIBLE_DEVICES=GPU-f75d3851-2a27-600b-ff53-be69abf1e506 # A100, device #3
    echo Device $gpu_id, UUID: $CUDA_VISIBLE_DEVICES
elif [ $gpu_id = 4 ]
then
    export CUDA_VISIBLE_DEVICES=GPU-ead507fb-4a3f-5203-5ba0-86f8a91905f1 # GTX2080Ti, local device
    echo Device $gpu_id, UUID: $CUDA_VISIBLE_DEVICES
else
    #export CUDA_VISIBLE_DEVICES= # Empty
    echo Error!! Device $gpu_id, UUID: $CUDA_VISIBLE_DEVICES
fi

nohup ${path}IBLBM.x \
      -iter_max 10000000 \
      -dump_frequency 10000 \
      -nx 2001 \
      -ny 1001 \
      -nz 1 \
      -u0 0.05 \
      -v0 0.0 \
      -nu 0.05 \
      -gpu_id ${dev_num} \
      > my.log 2>&1&

pid=$!
echo -n "dev_num " > save_pid.txt
echo  ${dev_num} >> save_pid.txt
echo -n "pid_num " >> save_pid.txt
echo  $pid >> save_pid.txt

#nvidia-smi -i ${dev_num}
#bash check.sh

more save_pid.txt
