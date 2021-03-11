#!/bin/bash
#SBATCH -J pynet50
#SBATCH -o pynet50_%j
#SBATCH -p gpu
#SBATCH --qos large
#SBATCH -N 1
#SBATCH -N 1
#SBATCH --gres=gpu:4

module load python
module load pytorch/gpu-1.6.0
module load cuda/102/toolkit/10.2.89  

CUDA_VISIBLE_DEVICES=0 nohup python3 ./main.py -a resnet50  --resume ./checkpoint/cp_resnet50.pth.tar ../../Datasets/ImageNet &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ./main.py -a adaptive_learn_resnet50.pth.tar -b 200 --resume ./checkpoint/cp_adaptive_learn_resnet50.pth.tar ../data/places/ &

../places365_standard/
 
 --epochs 150 --resume ./checkpoint/cp_resnet50.pth.tar 
 --resume ./checkpoint/cp_resnet50.pth.tar