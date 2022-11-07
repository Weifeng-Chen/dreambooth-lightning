export MODEL_NAME="../stable-diffusion-v1-4"
export INSTANCE_DIR="./weifeng"
export OUTPUT_DIR="./weifeng_output"
export CLASS_DIR="./person"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of weifeng person" \
  --class_prompt="a photo of person" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=100 \
  --max_train_steps=400 \
  # --with_prior_preservation --prior_loss_weight=1.0 \

  # --mixed_precision fp16


# srun accelerate launch --config_file ./accelerate.yaml train_dreambooth.py \
#   --pretrained_model_name_or_path=$MODEL_NAME  \
#   --instance_data_dir=$INSTANCE_DIR \
#   --output_dir=$OUTPUT_DIR \
#   --instance_prompt="photo of sks dog." \
#   --resolution=512 \
#   --train_batch_size=2 \
#   --gradient_accumulation_steps=1 \
#   --learning_rate=5e-6 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --num_train_epochs=50  \
#   # --max_train_steps=200 \
    # --center_crop

