CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --config config/fashion.yaml \
          --device_ids 0,1,2,3 --verbose --mode train --checkpoint 'log/fashion 10_07_21_15.34.03/00000249-cpk-reconstruction.pth' 
