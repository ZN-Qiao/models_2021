#!/bin/bash
#SBATCH -J .5resnet
#SBATCH -o .5resnet_%j
#SBATCH -p gpu
#SBATCH --qos large
#SBATCH -N 1
#SBATCH -N 1
#SBATCH --gres=gpu:4




CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./main.py -a resnet50 ../places365_standard/
