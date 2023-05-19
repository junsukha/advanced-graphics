import torch
import numpy as np
from rendering import *

print("Loading model...")
# rf = RadianceField().cuda() # was .cuda()
PATH_renderer = 'renderer2.pth'
PATH_rf = 'rf2.pth'
# PATH_renderer = 'renderer2.pth'
# PATH_rf = 'rf2.pth'

# 1.5, 4.5
renderer = VolumeRenderer(near=1.5, far=4.5, n_samples=128, white_back=True, rand=False).cuda() # was .cuda()
renderer.load_state_dict(torch.load(PATH_renderer))
renderer.eval()

rf = RadianceField().cuda()
rf.load_state_dict(torch.load(PATH_rf))
rf.eval()

cam2world = np.load('cam2world.npy')
cam2world = torch.Tensor(cam2world).cuda()

images = np.load('images.npy')
images = torch.Tensor(images).cuda() # .cuda()
intrinsics = torch.tensor([[0.7, 0., 0.5],
                           [0., 0.7, 0.5],
                           [0., 0., 1.]]).cuda()

x_pix = get_opencv_pixel_coordinates(128, 128, device=device).cuda()
x_pix = x_pix.reshape(1, -1, 2)

print('Running test cases...')
with torch.no_grad():
    frames = []
    for i in range(len(cam2world)):
        model_in = {'cam2world': cam2world[i:i+1], 'intrinsics': intrinsics[None, ...], 'x_pix': x_pix}
        rgb, depth = renderer(model_in, rf)

        rgb = rgb.reshape(128, 128, 3).cpu().numpy()
        rgb *= 255
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        frames.append(rgb)

print('Saving final mp4...')
f = 'final2.mp4'
imageio.mimwrite(f, frames, fps=3, quality=7)