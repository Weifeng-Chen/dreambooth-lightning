export MODEL_NAME="../stable-diffusion-v1-4"
export INSTANCE_DIR="./new_concept_dog"
export OUTPUT_DIR="./dog"

python dreambooth_light.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog." \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_train_epochs=100  \
#   --max_train_steps=200 
