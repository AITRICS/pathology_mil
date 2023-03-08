dataset=tcga_stad
gpu=0
optimizer=sgd
lr_pretrained=ImageNet_Res50
scheduler=single
for milmodel in milmax milmean; do
    for lr_downstream in 1e-3 1e-4 1e-5 1e-6; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
        # optimizer=sgd
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
        # optimizer=adamw
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --mil-model $milmodel --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_milmodel_${milmodel}.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

        (( gpu+=1 ))
    done
done


for milmodel in Attention GatedAttention; do
    for lr_downstream in 1e-3 1e-4 1e-5 1e-6; do
        # lr mix 길이는 8이어야 함 (gpu 갯수와 매칭 되어야 함)
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_sgd.txt &        
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &
        # optimizer=sgd
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_sam.txt &
        # optimizer=adamw
        CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --mil-model $milmodel --dataset $dataset --pretrain-type $lr_pretrained --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler --pushtoken o.OsyxHt1pZuwUBoMEFYBuzHFNjV5ekr95 > LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}_milmodel_${milmodel}.txt &
        # CUDA_VISIBLE_DEVICES=$gpu nohup python train_mil_sam.py --dataset $dataset --pretrain-type ${lr_pretrained} --epochs 100 --optimizer $optimizer --lr $lr_downstream --weight-decay 1e-4 --scheduler $scheduler > ${lr_pretrained}_LR_${lr_downstream}_sch_${scheduler}_opt_${optimizer}.txt &

        (( gpu+=1 ))
    done
done