## ic --> instance classifier
## nc --> negative centroid


data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
# scheduler_centroid=single
dataset=CAMELYON16
# weight_cov=1.0
# optimizer_nc=adamw
process_num=4
workers=4

CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --optimizer-nc $optimizer_nc --process-num $process_num \
                                                --workers $workers --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
