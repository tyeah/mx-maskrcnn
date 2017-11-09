export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
export PYTHONPATH=incubator-mxnet/python/

TRAIN_DIR=model/res50-fpn/bottles/alternate6/
DATASET=Bottles
SET=train
TEST_SET=val

# Test

# python eval_maskrcnn.py \
#     --prefix ${TRAIN_DIR}/final \
#     --network resnet_fpn \
#     --has_rpn \
#     --dataset ${DATASET} \
#     --image_set ${TEST_SET} \
#     --result_path data/bottles/results/pred/ \
#     --epoch 0 \
#     --gpu 1

export CITYSCAPES_DATASET="data/bottles"
python data/cityscape/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py 
