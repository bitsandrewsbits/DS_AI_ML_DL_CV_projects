#!bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# Execute via accelerate:
accelerate launch diffusers/examples/text_to_image/train_text_to_image.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --snr_gamma=5.0 \
  --dataset_name="data/train2017/images" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="trained_S_D_model"
