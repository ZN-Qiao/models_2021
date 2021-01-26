#!/bin/bash
#SBATCH -J main1_in
#SBATCH -o main1_%j
#SBATCH -p gpu
#SBATCH --qos large
#SBATCH -N 1
#SBATCH -N 1
#SBATCH --gres=gpu:4

module load cuda/101/toolkit/10.1.243
module load cudnn/7.6.5/cuda101
module load pytorch/gpu-1.6.0

../places365_standard/ 
../data/places/
-b 128 --lr 0.05 
--resume ./checkpoint/best_cp_scnet50.pth.tar

CUDA_VISIBLE_DEVICES=0 nohup python3 /work/xm0036/zhinan/p3_relu_all/main.py -a adaptive_learn_resnet50 -b 128 --lr 0.05 --resume ./checkpoint/cp_adaptive_learn_resnet50.pth.tar ../places365_standard/ &
CUDA_VISIBLE_DEVICES=0 nohup python3 /work/xm0036/zhinan/dyrelu/main.py -a resnet50 -b 128 --lr 0.05 --resume ./checkpoint/learn_resnet50.pth.tar ../places365_standard/ &
