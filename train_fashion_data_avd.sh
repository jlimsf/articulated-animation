CUDA_VISIBLE_DEVICES=0 python run.py \
          --checkpoint 'log/fashion 22_06_21_21.13.47/00000099-cpk-reconstruction.pth' \
          --config 'config/fashion.yaml' \
          --device_ids 0 \
          --mode train_avd
