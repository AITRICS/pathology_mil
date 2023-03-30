
dataset=tcga_stad
gpu=0
optimizer=sgd
scheduler=single
milmodel=dsmil
for lr_pretrained in ImageNet_Res18_newstat ImageNet_Res50_newstat; do
    for lr_downstream in 1 1e-1 1e-2 1e-3; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
        # optimizer=sgd
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
        # optimizer=adamw
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam_test.py --mil-model $milmodel --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_milmodel_${milmodel}_prt_${lr_pretrained}.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

        (( gpu+=1 ))
    done
done

gpu=0
optimizer=adamw
for lr_pretrained in ImageNet_Res18_newstat ImageNet_Res50_newstat; do
    for lr_downstream in 1e-1 1e-2 1e-3 1e-4; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
        # optimizer=sgd
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
        # optimizer=adamw
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam_test.py --mil-model $milmodel --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_milmodel_${milmodel}_prt_${lr_pretrained}.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

        (( gpu+=1 ))
    done
done