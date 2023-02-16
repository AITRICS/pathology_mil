# dataset=CAMELYON16
# gpu=0
# for lr_pretrained in simclr_lr1e-1 simclr_lr1; do
#     for lr_downstream in 1e-1 1e-2 1e-3 1e-4; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd.txt &
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer adamw --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_adamw.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_swa.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_swa.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_sam.txt &

#         (( gpu+=1 ))
#     done
# done

# dataset=tcga_lung
# gpu=0
# for lr_pretrained in simclr_lr1e-1 simclr_lr1; do
#     for lr_downstream in 1e-1 1e-2 1e-3 1e-4; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer adamw --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_adamw.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_swa.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_swa.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_sam.txt &

#         (( gpu+=1 ))
#     done
# done

dataset=tcga_stad
gpu=0
for lr_pretrained in simclr_lr1e-1 simclr_lr1; do
    for lr_downstream in 1e-1 1e-2 1e-3 1e-4; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd.txt &
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer adamw --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_adamw.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_swa.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_swa.txt &
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_sam.txt &

        (( gpu+=1 ))
    done
done

