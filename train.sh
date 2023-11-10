CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config_file='train.yaml'  2>&1 | tee train.log
