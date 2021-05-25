CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python3 ./main.py -a resnet101 --resume ./checkpoint/cp_resnet101.pth.tar ../data/places &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python3 ./main.py -a resnet101 --resume ./checkpoint/best_cp_resnet101.pth.
tar ../data/places &

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python3 ./main.py -a resnet101 ../data/places &