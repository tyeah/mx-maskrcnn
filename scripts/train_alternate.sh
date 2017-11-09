export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=incubator-mxnet/python/

TRAIN_DIR=model/res50-fpn/bottles/alternate10/
DATASET=Bottles
DATASET_PATH=data/bottles
SET=train
TEST_SET=val
mkdir -p ${TRAIN_DIR}

# Train
python train_alternate_mask_fpn.py \
    train \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --prefix ${TRAIN_DIR} \
    --dataset_path ${DATASET_PATH} \
    --rpn_lr 0.0001 \
    --rcnn_lr 0.0001 \
    --gpu 0 |& tee -a ${TRAIN_DIR}/train.log

