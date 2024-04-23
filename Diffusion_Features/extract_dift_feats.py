import argparse
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
import pathlib
import os
import numpy as np
import argparse

def zero_padding(inputs, pad_h, pad_w):
    
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result

parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', nargs='+', type=str, default='data',
                    help='''path to the images''')
parser.add_argument('--t', default=0, type=int, 
                    help='time step')
parser.add_argument('--s', default=0, type=int, 
                    help='scale')
args = parser.parse_args()

input_folder = args.data
dift = SDFeaturizer('stabilityai/stable-diffusion-2-1')
up_ft_index = args.s
t = args.t
output_folder = os.path.join(input_folder, f'feats_{up_ft_index}_{t}')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
image_paths = sorted(list(pathlib.Path(input_folder).glob('*.png')))
for i, path in enumerate(image_paths):
    img = Image.open(path).convert('RGB') 
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2  
    # img_tensor = zero_padding(img_tensor.permute(1, 2, 0).numpy(), 376, 1242) 
    # print((img_tensor).shape)
    ft = dift.forward(img_tensor,
                        t=t,
                        up_ft_index=up_ft_index,
                        ensemble_size=8)
    output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(path))[0] + '.pt')
    ft = torch.save(ft.squeeze(0).cpu(), output_path) # save feature in the shape of [c, h, w]
    print(f'saved_{i}')


