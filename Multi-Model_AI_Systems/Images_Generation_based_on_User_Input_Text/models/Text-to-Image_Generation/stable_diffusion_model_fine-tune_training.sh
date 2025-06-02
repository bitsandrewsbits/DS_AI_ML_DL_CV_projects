#!bin/bash
export HF_ENDPOINT=https://hf-mirror.com

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --snr_gamma=5.0 \
  --dataset_name="data" \
  --resolution=512 --center_crop --random_flip \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_samples=1000 \
  --train_batch_size=20 \
  --max_train_steps=10 \
  --total_train_epochs=100 \
  --learning_rate=1e-04 \
  --max_grad_norm=5 \
  --enable_xformers_memory_efficient_attention \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --validation_prompt="Black Lamborghini at midnight city." \
  --val_batch_size=20 \
  --validation_epochs=1 \
  --report_to="tensorboard" \
  --output_dir="finetuned_S_D_model"
