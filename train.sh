gpu=0
data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
scheduler_centroid=None
dataset=CAMELYON16
train_instance=intrainstance_vc
ic_num_head=5
ic_depth=2
weight_agree=1.0
weight_disagree=1.0
weight_cov=1.0
stddev_disagree=1.0
optimizer_nc=sgd
lr=0.003
lr_center=0.0001
mil_model=Dtfd

for weight_cov in 0 1; do
    for lr in 0.01 0.003 0.001 0.0003; do

        CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
        --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
        --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &

        (( gpu+=1 ))
    done
done
