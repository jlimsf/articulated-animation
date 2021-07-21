python run.py \
      --checkpoint 'finetuning/ubc_full_path.pth' \
      --config 'config/fashion_frames_only.yaml' \
      --device_ids 0,1,2,3 \
      --mode train_customer_data

aws s3 cp finetuning/train-vis/ s3://cqdatascience/john/train-vis/ --recursive
