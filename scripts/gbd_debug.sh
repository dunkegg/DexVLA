#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1   # 让 CUDA 异步调用变同步，方便调试

# 下面直接用 python 运行，不用 torchrun 方便 attach 和调试
gdb --args python ./train_vla.py \
  --use_reasoning False \
  --lora_enable False \
  --using_film True \
  --action_dim 3 \
  --state_dim 3 \
  --flash_attn False \
  --chunk_size 30 \
  --load_pretrain_dit False \
  --policy_head_type scale_dp_policy \
  --policy_head_size "ScaleDP_H" \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --history_images_length 9 \
  --task_name debug \
  --model_name_or_path checkpoints/qwen2_vl \
  --version v0 \
  --enable_distilbert False \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir OUTPUT/qwen2_follow_test \
  --max_steps 80000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy steps \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type cosine \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 1 \
  --dataloader_persistent_workers False \
  --lazy_preprocess True \
  --policy_class scale_dp_policy \
  --concat token_cat \
  --report_to tensorboard \
  --logging_dir OUTPUT/qwen2_follow_test/log
