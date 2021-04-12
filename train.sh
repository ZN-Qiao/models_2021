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

CUDA_VISIBLE_DEVICES=1 nohup python3 ./main.py -a resnet50 -b 512 --resume ./checkpoint/cp_resnet50.pth.tar ../../Datasets/ImageNet &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ./main.py -a adaptive_learn_resnet50.pth.tar -b 200 --resume ./checkpoint/cp_adaptive_learn_resnet50.pth.tar ../data/places/ &

../places365_standard/
 
--epochs 150 --resume ./checkpoint/cp_resnet50.pth.tar 

--resume ./checkpoint/cp_resnet50.pth.tar
 
CUDA_VISIBLE_DEVICES=1 python3 ./main.py -a resnet50 -b 512 --resume ./checkpoint/cp_resnet50.pth.tar ../../Datasets/ImageNet

CUDA_VISIBLE_DEVICES=0 nohup python3 ./main.py -a resnet50 -b 512 --resume ./checkpoint/cp_resnet50.pth.tar ../../Datasets/ImageNet &

CUDA_VISIBLE_DEVICES=0 nohup python3 ./main.py -a resnet50 -b 300 --resume ./checkpoint/cp_resnet50.pth.tar ../../Datasets/ImageNet &

CUDA_VISIBLE_DEVICES=0,1 nohup python3 ./main.py -a resnet50 -b 512 --resume ./checkpoint/cp_resnet50.pth.tar ../imagenet &

CUDA_VISIBLE_DEVICES=0,1 nohup python3 ./main.py -a resnet50 -b 512 ../imagenet &

CUDA_VISIBLE_DEVICES=0,1 nohup python3 ./main.py -a resnet50 -b 512 ../data/imagenet &

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./main.py -a resnet50 --num_classes 1000 --resume ./checkpoint/cp_resnet50.pth.tar --epochs 150 ../ImageNet/

nohup python3 ./main.py -a resnet50 --num_classes 1000 --resume ./checkpoint/cp_resnet50.pth.tar --epochs 150 ../data/imagenet &

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./main.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar --epochs 150 ../ImageNet/

nohup python3 ./main_larger.py -a resnet101 ../data/places &

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ./main.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar --epochs 150 ../places365_standard/

nohup python3 ./main.py -a resnet101 ../data/places &

nohup python3 ./main_small.py -a resnet50 ../places365_standard &

nohup python3 ./main_small.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar ../places365_standard &

python3 ./main_blur.py -a resnet50 -e ../../data/imagenet

python3 ./main_blur.py -a resnet50 -e ../../data/places

nohup python3 ./main.py -a resnet101 --resume ./checkpoint/cp_resnet101.pth.tar ../data/places &

python3 ./main.py -a resnet50 --resume ./checkpoint/places.pth.tar ../data/places

python3 ./main.py -a resnet50 -b 128 --lr 0.05 --resume ./checkpoint/places.pth.tar ../data/places

python3 ./main_fft.py -a resnet50 -e ../../data/imagenet

nohup python3 ./main.py -a resnet50 -b 128 --lr 0.05 ../data/places &

python3 ./main_fft.py -a resnet50 -e ../../imagenet

python3 ./main_fft.py -a resnet50 -e --bw 30 --filter 1 ../../imagenet

python3 ./main_blur.py -a resnet50 -e ../../places365_standard

nohup python3 ./main.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar ../data/places &

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 ./main.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar ../data/places &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 ./main_imagenet.py -a PreActResNet50 ../../../data/places &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 ./main_imagenet.py  ../../../data/places

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 ./main_imagenet.py  ../../../data/places/data100/ &

python3 ./main_imagenet.py -a PreActResNet50 ../../../places365_standard

python3 ./main.py -a PreActResNet50 ../../../places365_standard

python3 ./main_imagenet.py -a PreActResNet50 ../../../imagenet

python3 ./main_imagenet.py  ../../data/imagenet/data100/

python3 ./main.py  ../../../data/places/data100/

nohup python3 ./main.py ../../../data/places/data100/ &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 main_imagenet.py -b 128 --lr 0.05 ../../../data/places/data100/ &

nohup python3 ./main.py -a resnet50 ../../../data/places/data100/ &

nohup python3 ./main.py -b 128 --lr 0.05 ../../../data/places/data100/ &

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 ./main_imagenet.py  -a resnet50 ../data/places/data100/ &

python3 ./main_imagenet.py -a resnet50 --num_classes 100 ../../../data/imagenet/data100/

nohup python3 ./main_imagenet.py -a resnet50 --num_classes 100 ../../../data/imagenet/data100/ &

nohup python3 ./main.py -a resnet50 --num_classes 100 ../../../data/places/data100/ &

nohup python3 ./main.py -a resnet50 --num_classes 100 ../../../data/places/data100/ &

nohup python3 ./main_imagenet.py -a resnet50 --num_classes 100 ../../../data/imagenet/data100/ &

nohup python3 ./main.py -a resnet50 --num_classes 100 ../../../data/places/data100/ &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python3 ./main_imagenet.py -a resnet50 --num_classes 100 ../../../data/imagenet/data100/ &

nohup python3 ./main_imagenet.py -a wide_resnet50_2 --num_classes 100 ../../../data/imagenet/data100/ &

nohup python3 ./main.py -a wide_resnet50_2 --num_classes 100 ../../../data/places/data100/ &

python3 ./main.py -a resnet50 --num_classes 100 ../../../data/places/data100/

python3 ./main100.py -a resnet50 --num_classes 100 ../../../data/places/data100/

nohup python3 ./main100.py -a resnet50 --num_classes 100 ../../../data/places/data100/ &

nohup python3 ./main.py -a resnet101 --num_classes 365 ../data/places/&

nohup python3 ./main100.py -a resnet50 --num_classes 100 ../../../data/imagenet/data100/ &

nohup python3 ./main100.py -a resnet50 --num_classes 100 --resume ./checkpoint/cp_resnet50.pth.tar ../../data/imagenet/data100/ &

nohup python3 ./main100.py -a resnet50 --num_classes 100 --resume ./checkpoint/cp_resnet50.pth.tar ../../data/imagenet/data100/ &

nohup python3 ./main100.py -a resnet50 --num_classes 100 --resume ./checkpoint/cp_resnet50.pth.tar ../../data/places/data100/ &

nohup python3 ./main.py -a resnet101  -b 128 --lr 0.05 --resume ./checkpoint/cp_resnet101.pth.tar ../data/places &

nohup python3 ./main_larger.py -a resnet50 --resume ./checkpoint/cp_resnet50.pth.tar ../data/places &

