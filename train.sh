gpu=0
dataset=CAMELYON16
data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
lr_downstream=0.003

milmodel=Dtfd_noscale
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mil-model $milmodel --dataset $dataset --data-root $data_root --epochs 200 --workers 4 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
(( gpu+=1 ))

milmodel=Dtfd_scale
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mil-model $milmodel --dataset $dataset --data-root $data_root --epochs 200 --workers 4 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
(( gpu+=1 ))

milmodel=Dtfd_add_scale
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mil-model $milmodel --dataset $dataset --data-root $data_root --epochs 200 --workers 4 --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
(( gpu+=1 ))


milmodel=Dtfd_tune
aux_loss=loss_div_vc
# aux_loss=loss_jsd
# aux_loss=loss_vc
# aux_head=2
# layernum_head=2
weight_agree=1.0
weight_disagree=1.0
weight_cov=1.0
stddev_disagree=1.0

for layernum_head in 1 2; do
    for lr_downstream in 0.003 0.0003; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
        # optimizer=sgd
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
        # optimizer=adamw
        CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --layernum-head $layernum_head --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --mil-model $milmodel --dataset $dataset --data-root $data_root --epochs 200 --workers 4 --aux-loss $aux_loss --weight-cov $weight_cov --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

        (( gpu+=1 ))
    done
done
layernum_head=2
lr_downstream=0.00003
CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --layernum-head $layernum_head --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --mil-model $milmodel --dataset $dataset --data-root $data_root --epochs 200 --workers 4 --aux-loss $aux_loss --weight-cov $weight_cov --lr $lr_downstream --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
        