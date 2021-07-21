python demo.py --config 'config/fashion.yaml' \
              --checkpoint 'finetuning/ubc_full_path.pth' \
              --source_image '../clean_rebecca_taylor_frontals/abby-rebeccataylor--0b4f817e75f778121ac8051a0cd470d0.jpg' \
              --driving_video 'data/fashion_vids/test/91cC+1+C4SS.mp4' \
              --img_shape 256,256 \
              --mode standard \
              --result_video overfit.mp4

aws s3 cp overfit.mp4 s3://cqdatascience/john/
 # data/train/91-kqBbzDIS.mp4


 # --source_image 'data/fashion_png/test/91bxAN6BjAS/00001.png' \
 # --driving_video 'data/fashion_vids/test/91-3003CN5S.mp4' \

# log/fashion 09_07_21_13.08.37

# ../clean_rebecca_taylor_frontals/abby-rebeccataylor--0b4f817e75f778121ac8051a0cd470d0.jpg
