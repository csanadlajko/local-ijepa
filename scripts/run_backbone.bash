echo "executing backbone training (40ep)"

python3 local_jepa.py \
    --dataset coco \
    --coco_train_folder /mnt/c/Users/csanadlajko/Documents/suli/tdk/train2017/train2017/ --coco_train_annotation /mnt/c/Users/csanadlajko/Documents/suli/tdk/annotations_trainval2017/annotations/instances_train2017.json \
    --result_folder "/home/clajko/suli/tdk/local-ijepa/results/euclidean" \
    --image_size 256 \
    --patch_size 16 \
    --embed_dim 256 \
    --mlp_dim 512 \
    --depth 6 \
    --num_heads 8 \
    --num_target 4 \
    --batch_size 16 \
    --epochs 40 \
    --lr 0.0003 \
    --finetune_lr 0.001 \
    --num_classes 80 \
    --num_block_categories 81 \
    --multimodal_run n \
    --debug y \
    --geometry euclidean \
    --skip_finetune y
