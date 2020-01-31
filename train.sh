MODEL=$1
DATASET=$2
LOSS_TYPE=$3
BATCH_SIZE=$4
RESIZE=$5
REFINE_NETWORK=${6:-0}
SECOND_LOSS=${7:-0}

python main.py --model $MODEL --dataset $DATASET --loss_type $LOSS_TYPE --batch_size $BATCH_SIZE \
               --resize $RESIZE --epochs 100 --norm_layer batch --debug 0 \
               --refine_network $REFINE_NETWORK --second_loss $SECOND_LOSS