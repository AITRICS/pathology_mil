
# gpu=2
# dataset=CAMELYON16
# milmodel=Dtfd
# for data_root in /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_new_pkl/ImageNet_Res50_newstat /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_transmil_pkl/ImageNet_Res50_newstat; do
#     for lr_downstream in 0.03 0.01 0.003 0.001; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         # optimizer=sgd
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
#         # optimizer=adamw
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 200 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

#         (( gpu+=1 ))
#     done
#     gpu=2
# done


# gpu=2
# dataset=CAMELYON16
# milmodel=Dsmil
# for data_root in /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_new_pkl/ImageNet_Res50_newstat /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_transmil_pkl/ImageNet_Res50_newstat; do
#     for lr_downstream in 0.03 0.01 0.003 0.001; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         # optimizer=sgd
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
#         # optimizer=adamw
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mil-model $milmodel --dataset $dataset --epochs 40 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

#         (( gpu+=1 ))
#     done
    # gpu=2
# done


gpu=0
dataset=CAMELYON16
milmodel=MilTransformer
data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
for if_learn_instance in True False; do
    for lr_downstream in 0.0003 0.0001 0.00003 0.00001; do 
        CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 100 --lr $lr_downstream --if-learn-instance $if_learn_instance --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
        (( gpu+=1 ))        
    done
done

# gpu=2
# dataset=CAMELYON16
# milmodel=Attention
# for data_root in /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_new_pkl/ImageNet_Res50_newstat /mnt/aitrics_ext/ext01/shared/camelyon16_jpeg_new_pkl/camelyon16_jpeg_transmil_pkl; do
#     for lr_downstream in 0.003 0.001 0.0003 0.0001; do    
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         # optimizer=sgd
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
#         # optimizer=adamw
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mil-model $milmodel --dataset $dataset --epochs 100 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         (( gpu+=1 ))
#        
#     done
    # gpu=2
# done

# dtfd (200ep), dsmil (40ep): 0.01 작
# Att (100ep): 0.001 작은쪽
# Gated ATT (100ep): 0.003 위