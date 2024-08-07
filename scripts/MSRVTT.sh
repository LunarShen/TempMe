DATA_PATH=Your_Path
OUTPUT_PATH=log/MSRVTT
PRETRAINED_PATH=Your_Path

python -m torch.distributed.launch --nproc_per_node=4 main.py \
  --do_train 1 --workers 8 \
  --anno_path ${DATA_PATH} --video_path ${DATA_PATH}/Videos --datatype msrvtt \
  --output_dir ${OUTPUT_PATH} \
  --pretrained_path ${PRETRAINED_PATH}
