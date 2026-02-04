#!/bin/bash

set -e

LLM=qwen2_vl
LLM_MODEL_SIZE=2B

ACTION_HEAD=scale_dp_policy  #unet_diffusion_policy or scale_dp_policy

DIT_PRETRAIN=checkpoints/ScaleDP/open_scale_dp_h_backbone.ckpt
MNOP=checkpoints/MiMo-Embodied-7B # official qwen2_vl weights
TASKNAME=follow_pixel

OUTPUT=OUTPUT/qwen2_5_single_follow_pixel_coord_loss
JSON_PATH=data/split_data/single_follow_pixel/norm_status.json
mkdir -p $OUTPUT

deepspeed --master_port 29604 --include=localhost:0,1,2,3 ./train_qwen2_5_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --lora_enable False \
  --using_film True \
  --action_dim 3 \
  --state_dim 3 \
  --flash_attn True \
  --chunk_size 30 \
  --load_pretrain_dit False \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_H" \
  --image_size_stable "(320,180)" \
  --image_size_wrist "(320,180)" \
  --history_images_length 0 \
  --task_name ${TASKNAME} \
  --model_name_or_path $MNOP \
  --version v0 \
  --enable_distilbert False \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir $OUTPUT \
  --norm_json_path $JSON_PATH \
  --max_steps 10000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 1 \
  --dataloader_persistent_workers False \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log | tee $OUTPUT/log.log


status=${PIPESTATUS[0]}  # deepspeed 的退出码

if [ $status -ne 0 ]; then
    echo "❌ Deepspeed 崩了，退出码：$status"
    exit $status
fi
for dir in "$OUTPUT"/*/ ; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo $OUTPUT
