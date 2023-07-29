dataset=CAMELYON16
milmodel=Dtfd_tune
aux_loss=loss_divdis
aux_head=1
# weight_cov=1
data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
auxloss_weight=1 # default
weight_cov=1 # default

gpu=5
lr_downstream=0.01
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 200 --workers 4 --aux-loss $aux_loss --aux-head $aux_head --weight-cov $weight_cov --lr $lr_downstream --auxloss-weight $auxloss_weight > ./nohups/WC_${weight_cov}_LR_${lr_downstream}_AH_${aux_head}_AW_${auxloss_weight}_milmodel_${milmodel}_gpu_${gpu}.txt &


gpu=6
lr_downstream=0.001
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 200 --workers 4 --aux-loss $aux_loss --aux-head $aux_head --weight-cov $weight_cov --lr $lr_downstream --auxloss-weight $auxloss_weight > ./nohups/WC_${weight_cov}_LR_${lr_downstream}_AH_${aux_head}_AW_${auxloss_weight}_milmodel_${milmodel}_gpu_${gpu}.txt &

gpu=7
lr_downstream=0.0001
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 200 --workers 4 --aux-loss $aux_loss --aux-head $aux_head --weight-cov $weight_cov --lr $lr_downstream --auxloss-weight $auxloss_weight > ./nohups/WC_${weight_cov}_LR_${lr_downstream}_AH_${aux_head}_AW_${auxloss_weight}_milmodel_${milmodel}_gpu_${gpu}.txt &


# lr_downstream=0.003
# for aux_head in 0; dog
#     for auxloss_weight in 1 0.1; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         # optimizer=sgd
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
#         # optimizer=adamw
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --mil-model $milmodel --dataset $dataset --epochs 200 --workers 4 --aux-loss $aux_loss --aux-head $aux_head --weight-cov $weight_cov --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 --auxloss-weight $auxloss_weight > ./nohups/WC_${weight_cov}_LR_${lr_downstream}_AH_${aux_head}_AW_${auxloss_weight}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
#         (( gpu+=1 ))
#     done
# done
