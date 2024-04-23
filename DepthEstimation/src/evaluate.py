import sys
sys.path.append('../')

import torch
from model import Model
from torch.utils.data import DataLoader
from datasets.kitti import KITTI_Train, KITTI_Test
from loss import get_depth_loss, resize_depth, si_log_loss
import os
import argparse
from PIL import Image
from torchvision.utils import save_image
from torchmetrics.regression import RelativeSquaredError
import math as mt

relative_squared_error = RelativeSquaredError().cuda()

def save_depth_image(depth, file_path, scale=1.0):
    depth = depth * scale * 256.0
    # depth_image = Image.fromarray(depth)
    # depth_image.save(file_path)
    save_image(depth, file_path, normalize=True)
    

parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', nargs='+', type=str, default='../data/images',
                    help='''path to the training data images folder''')
parser.add_argument('--anno_data', nargs='+', type=str, default='../data/annotations',
                    help='''path to the training data images folder''')
parser.add_argument('--chkt_path', default='../checkpoints/mse/ckpt_855.pt', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--save_path', default='../results/mixed_new', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--type', default='mixed', type=str, 
                    help='type of dift feature')
parser.add_argument('--ts', default=261, type=int, 
                    help='time step of dift feature')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.type == 'mixed':
    test_data = KITTI_Train(args.data, args.anno_data)
else:
    test_data = KITTI_Test(args.data, args.anno_data, args.ts)

test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

data = next(iter(test_dataloader))

model = Model().cuda()

model.load_state_dict(torch.load(args.chkt_path)['model_state_dict'], strict=False)
model.eval()
total_mse = 0.0
total_sqrel = 0.0
total_si = 0.0
for i, data in enumerate(test_dataloader):
    inputs, targets = data['features'], data['depth'].cuda() 
    for j in range(len(data['features'])):
        data['features'][j] = data['features'][j].cuda()

    pred_depth = model(data['features'])
    pred_depth = pred_depth.clamp(0,90).detach()
    # total_si += si_log_loss(resize_depth(pred_depth, targets.shape[2], targets.shape[3]), targets)
    total_mse += get_depth_loss(pred_depth, targets, type='mse').item()   
    total_sqrel += relative_squared_error(resize_depth(pred_depth, targets.shape[2], targets.shape[3]).squeeze(0).squeeze(0).flatten(), targets.squeeze(0).squeeze(0).flatten()).item()
    print(i, total_mse, total_sqrel)
    save_depth_image(pred_depth.cpu(), os.path.join(args.save_path, data['index'][0] + '.png'))

# total_si = total_si/len(test_dataloader)
total_mse = total_mse/len(test_dataloader)
total_sqrel = total_sqrel/len(test_dataloader)

print(f'sq_rel: {total_sqrel}, mse_loss: {total_mse}, rmse_loss: {mt.sqrt(total_mse)}')