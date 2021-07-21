def display(source, driving, generated=None):
    fig = plt.figure(figsize=(8 + 4 * (generated is not None), 6))

    ims = []
    for i in range(len(driving)):
        cols = [source]
        cols.append(driving[i])
        if generated is not None:
            cols.append(generated[i])
        im = plt.imshow(np.concatenate(cols, axis=1), animated=True)
        plt.axis('off')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=1000)
    plt.close()
    return ani


import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
from IPython.display import HTML

source_image = imageio.imread('../../Desktop/clean_rebecca_taylor_frontals/abby-rebeccataylor--0b4f817e75f778121ac8051a0cd470d0.jpg')
driving_video = imageio.mimread('data/fashion_png/test/out.mp4', memtest=False)
predictions = imageio.mimread('../../Downloads/overfit (37).mp4', memtest=False)

source_image = resize(source_image, (384, 384))[..., :3]
driving_video = [resize(frame, (384, 384))[..., :3] for frame in driving_video]
predictions = [resize(frame, (384, 384))[..., :3] for frame in predictions]

HTML(display(source_image, driving_video, predictions).to_html5_video())
