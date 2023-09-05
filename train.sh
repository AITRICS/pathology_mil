gpu=3
#camerelon: /mnt/aitrics_ext/ext01/shared/camelyon16_eosin_224_16_pkl_0524/swav_res50
#lung: /mnt/aitrics_ext/ext01/shared/tcgalung_eosin_224_16_pkl_0806/swav_res50/ -> Old
#lung: /mnt/aitrics_ext/ext01/shared/tcgalung_dsmil
data_root=/mnt/aitrics_ext/ext01/shared/tcgalung_dsmil
# single도 해볼 수 있음
scheduler_centroid=single
# tcga_lung, CAMELYON16 
dataset=tcga_lung
# None, intrainstance_divdis, interinstance_vc, interinstance_cosine, intrainstance_vc, intrainstance_cosine, semisup1
train_instance=semisup1 
# intrainstance_divdis 하면 당연히 ic_num_head 1
ic_num_head=2
# intrainstance_divdis 하면 ic_num_head 1
ic_depth=1
weight_agree=1.0
weight_disagree=1.0
# 0도 해봄직
weight_cov=1.0
stddev_disagree=1.0
lr=0.003
lr_center=0.00001
# Dtfd, Attention, GatedAttention, Dsmil 가능
mil_model=Dsmil
alpha=0.1
beta=0
optimizer_nc=adamw

    # for lr in 0.01 0.003 0.001 0.0003; do
# for lr in 0.01 0.003; do


#     CUDA_VISIBLE_DEVICES=$gpu python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
#     --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --stddev-disagree $stddev_disagree \
#     --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#     (( gpu+=1 ))
# done


for lr in 0.0001; do
    for alpha in 0.1 0.2 0.3; do
        CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
        --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --stddev-disagree $stddev_disagree \
        --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --alpha $alpha > logs/DSET_${dataset}_milmodel_${mil_model}_alpha_${alpha}_train_instance_${train_instance}_gpu_${gpu}.txt &
        (( gpu+=1 ))
    done
done
weight_cov=1.0
weight_disagree=0.0
for weight_agree in 0.01 0.1; do
    # 주의! weight_disagree 가 두번 사용되었음 !!!
    CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
    --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
    --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v --save $save\
    --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
    (( gpu+=1 ))
done



# lr_center=0.1

# for weight_agree in 3.0 0.3 1.0; do
#     for weight_disagree in 10.0 3.0 1.0 0.3; do
#         # 주의! weight_disagree 가 두번 사용되었음 !!!
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --data-root $data_root --scheduler-centroid $scheduler_centroid --dataset $dataset --train-instance $train_instance \
#         --ic-num-head $ic_num_head --ic-depth $ic_depth --weight-agree $weight_agree --weight-disagree $weight_disagree --weight-cov $weight_cov --stddev-disagree $stddev_disagree \
#         --optimizer-nc $optimizer_nc --lr $lr --lr-center $lr_center --mil-model $mil_model --passing-v $passing_v --save $save\
#         --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > DTFDprecompu_LR_${lr_downstream}_milmodel_${milmodel}_gpu_${gpu}.txt &
#         (( gpu+=1 ))
#     done
# done

