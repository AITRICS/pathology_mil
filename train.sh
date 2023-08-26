gpu=0
data_root=/mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
# maximoff
# data_root=/mnt/aitrics_ext/ext02/camelyon16_eosin_224_16_pkl_0524/swav_res50
# data_root=/mnt/aitrics_ext/ext01/shared/tcgalung_dsmil
# maximoff
# data_root=/mnt/aitrics_ext/ext02/shared/tcgalung_dsmil
# single도 해볼 수 있음
scheduler_centroid=single
# dataset=tcga_lung
dataset=CAMELYON16
# None, intrainstance_divdis, interinstance_vc, interinstance_cosine, intrainstance_vc, intrainstance_cosine
train_instance=interinstance_vi
# intrainstance_divdis 하면 당연히 ic_num_head 1
ic_num_head=1
# intrainstance_divdis 하면 ic_num_head 1
ic_depth=1
weight_agree=10.0
weight_disagree=10.0
# 0도 해봄직
weight_cov=0.0
stddev_disagree=1.0
# negative centroid 업데이트 방법: sgd, adamw, adam
optimizer_nc=adamw
lr=0.003
lr_center=0.00001
# Dtfd, Dsmil, Attention, GatedAttention 가능
mil_model=Dsmil
passing_v=1

for lr_center in 0.0001 0.00001; do
    for weight_disagree in 0.3 1.0 3.0 10.0; do
        # 주의! weight_disagree 가 두번 사용되었음 !!!
        CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
        --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_disagree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
        --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v \
        --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &

        (( gpu+=1 ))
    done
done

# # 젤 잘나온 lr_center, weight 주변까지 3*3개
# passing_v
# lr_center=0.00001
# for weight_agree in 0.0001 0.00001; do
#     for weight_disagree in 0.3 1.0 3.0 10.0; do
#         # 주의! weight_disagree 가 두번 사용되었음 !!!
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
#         --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
#         --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v \
#         --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &

#         (( gpu+=1 ))
#     done
# done

# # 젤 잘나온 weight사용, lr_center 주변까지 3개
# passing_v
# stddev_disagree=1.0
# weight_agree
# weight_disagree
# for lr_center in 0.0001 0.00003 0.00001; do
#     for stddev_disagree in 0.5 5.0 1.0; do
#         # 주의! weight_disagree 가 두번 사용되었음 !!!
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
#         --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
#         --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v \
#         --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &

#         (( gpu+=1 ))
#     done
# done