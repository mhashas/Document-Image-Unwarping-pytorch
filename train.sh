MODEL=$1
DATASET=$2
LOSS_TYPE=$3
BATCH_SIZE=$4
GAN_TRAINING=$5
RESIZE=$6
REFINE_NETWORK=${7:-0}
SECOND_LOSS=${8:-0}

python main.py --model $MODEL --dataset $DATASET --loss_type $LOSS_TYPE --batch_size $BATCH_SIZE \
               --resize $RESIZE --gan_training $GAN_TRAINING --epochs 100 --norm_layer batch --debug 0 \
               --refine_network $REFINE_NETWORK --second_loss $SECOND_LOSS