#!/bin/bash
#SBATCH -J pynet50
#SBATCH -o pynet50_%j
#SBATCH -p gpu
#SBATCH --qos large
#SBATCH -N 1
#SBATCH -N 1
#SBATCH --gres=gpu:4

module load python/3.7.4
module load pytorch/gpu-1.5.0
module load cuda/101/toolkit/10.1.243

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ./main.py -a adaptive_learn_resnet50 -b 128 --lr 0.05 ../data/places/ &


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ./main.py -a resnet50 --resume ./checkpoint/best_cp_resnet50.pth.tar ../places365_standard/ &

../../Datasets/ImageNet  
 
 ../places365_standard/
 