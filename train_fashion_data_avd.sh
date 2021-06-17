CUDA_VISIBLE_DEVICES=0 python run.py \
          --checkpoint 'log/fashion 16_06_21_00.46.15/00000029-cpk-reconstruction.pth' \
          --config 'config/fashion.yaml' \
          --device_ids 0 \
          --mode train_avd
