gpu=0
scheduler_centroid=single
train_instance=interinstance_vic
ic_num_head=1
ic_depth=1
weight_cov=0.0
optimizer_nc=adamw
passing_v=1

########################################################
stddev_disagree=1.5


# data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
# maximoff
data_root=/home/chris/storage/camelyon16_eosin_224_16_pkl_0524/swav_res50

# data_root=/home/chris/storage/tcgalung_dsmil
# maximoff
# data_root=/mnt/aitrics_ext/ext02/shared/tcgalung_dsmil

# dataset=tcga_lung
dataset=CAMELYON16

weight_agree=1.0
weight_disagree=0.3

lr=0.0003
lr_center=0.0003
# Dtfd, Dsmil, Attention, GatedAttention 가능
mil_model=Dsmil


for lr_center in 0.000003 0.00001 0.00003 0.0001 0.0003; do
    # 주의! weight_disagree 가 두번 사용되었음 !!!
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
    --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
    --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v \
    --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
    (( gpu+=1 ))
done



# lr_center=0.1

# for weight_agree in 3.0 0.3 1.0; do
#     for weight_disagree in 10.0 3.0 1.0 0.3; do
#         # 주의! weight_disagree 가 두번 사용되었음 !!!
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
#         --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
#         --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v \
#         --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         (( gpu+=1 ))
#     done
# done

