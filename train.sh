dataset=CAMELYON16
# gpu=0
# for lr_pretrained in simclr_lr10 simclr_lr30; do
#     for lr_downstream in 1e-4 1e-5 1e-6 1e-7; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd.txt &
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer adamw --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_adamw_restart.txt &
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_swa.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_swa_restart.txt &
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_sam_restart.txt &

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

# dataset=tcga_stad
# gpu=0
# for lr_pretrained in simclr_lr1e-1 simclr_lr10; do
#     for lr_downstream in 1e-2 1e-3 1e-4 1e-5; do
#         # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd.txt &
#         CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer adamw --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_adamw.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_swa.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_swa.txt &
#         # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer sgd --lr $lr_downstream --weight-decay 1e-4 > ${lr_pretrained}_LR_${lr_downstream}_sgd_sam.txt &

#         (( gpu+=1 ))
#     done
# done


CUDA_VISIBLE_DEVICES=0 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.001 --weight-decay 1e-4 --workers 2 > simclr_lr0001_LR_01_sgd_restart.txt &
CUDA_VISIBLE_DEVICES=0 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer adamw --lr 0.001 --weight-decay 1e-4 --workers 2 > simclr_lr0001_LR_01_adamw_restart.txt &

CUDA_VISIBLE_DEVICES=1 nohup python train_mil_swa.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.001 --weight-decay 1e-4 --workers 2 > simclr_lr0001_LR_001_sgd_swa_restart.txt &
CUDA_VISIBLE_DEVICES=1 nohup python train_mil_sam.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.001 --weight-decay 1e-4 --workers 2 > simclr_lr0001_LR_001_sgd_sam_restart.txt &

CUDA_VISIBLE_DEVICES=2 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.0001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_00001_sgd_restart.txt &
CUDA_VISIBLE_DEVICES=2 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer adamw --lr 0.0001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_00001_adamw_restart.txt &
CUDA_VISIBLE_DEVICES=3 nohup python train_mil_swa.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.0001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_00001_sgd_swa_restart.txt &
CUDA_VISIBLE_DEVICES=3 nohup python train_mil_sam.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.0001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_00001_sgd_sam_restart.txt &

CUDA_VISIBLE_DEVICES=4 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.00001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_000001_sgd_restart.txt &
CUDA_VISIBLE_DEVICES=4 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer adamw --lr 0.00001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_000001_adamw_restart.txt &
CUDA_VISIBLE_DEVICES=5 nohup python train_mil_swa.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.00001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_000001_sgd_swa_restart.txt &
CUDA_VISIBLE_DEVICES=5 nohup python train_mil_sam.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.00001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_000001_sgd_sam_restart.txt &

CUDA_VISIBLE_DEVICES=6 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.000001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_0000001_sgd_restart.txt &
CUDA_VISIBLE_DEVICES=6 nohup python train_mil.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer adamw --lr 0.000001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_0000001_adamw_restart.txt &
CUDA_VISIBLE_DEVICES=7 nohup python train_mil_swa.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.000001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_0000001_sgd_swa_restart.txt &
CUDA_VISIBLE_DEVICES=7 nohup python train_mil_sam.py --dataset $dataset --pretrain-type simclr_lr10 --epochs 100 --optimizer sgd --lr 0.000001 --weight-decay 1e-4 --workers 2 > simclr_lr10_LR_0000001_sgd_sam_restart.txt &
