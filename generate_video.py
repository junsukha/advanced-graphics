import imageio
import cv2
import numpy as np
frames = []
for i in range(20):
    path = 'images/image{:05d}.png'.format(100*i)
    image = cv2.imread(path, 3)
    # print(image.shape)
    # exit(0)
    # model_in = {'cam2world': cam2world[i:i+1], 'intrinsics': intrinsics[None, ...], 'x_pix': x_pix}
    # rgb, depth = renderer(model_in, rf)

    # rgb = rgb.reshape(128, 128, 3).cpu().numpy()
    # rgb *= 255
    # rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    frames.append(np.array(image))

print('Saving final mp4...')
f = 'training.mp4'
imageio.mimwrite(f, frames, fps=1, quality=7)