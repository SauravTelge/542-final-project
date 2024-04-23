import torch
from model import Model
from torch.utils.data import DataLoader
from kitti_dataset import KITTI, KITTI_Test
from Sintel_dataset import make_dataset, ListDataset, make_dataset_eval
from loss import get_flow_loss, EPE, resize_flow2d, realEPE, mse_loss
from utils import save_flow_png
import os
import argparse
from torch.nn.functional import interpolate
from tqdm import trange, tqdm


parser = argparse.ArgumentParser(
        description='''Train U-Net for Optical Flow.''')
    
parser.add_argument('--data', nargs='+', type=str, default='data',
                    help='''path to the training data images folder''')
parser.add_argument('--chkt_path', default='checkpoints_OF_mse/ckpt_305.pt', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--save_path', default='../results/single_261', type=str, 
                    help='path to save checkpoints')
parser.add_argument('--dataset', default='kitti', type=str, 
                    help='dataset')
parser.add_argument('--type', default='single', type=str, 
                    help='type of dift feature')
parser.add_argument('--ts', default=261, type=int, 
                    help='time step of dift feature')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if (args.dataset == 'kitti') and (args.type == 'mixed'):
    test_data = KITTI(args.data)
elif args.dataset == 'kitti':
    test_data = KITTI_Test(args.data, args.ts)
else:
    feature_list, flow_list = make_dataset_eval(args.data, dataset_type="clean")
    test_data = ListDataset(args.data, feature_list, flow_list)


test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)


model = Model().cuda()
model.load_state_dict(torch.load(args.chkt_path)['model_state_dict'], strict=True)
model.eval()

total_mse = 0.0
total_epe = 0.0
average_EPE = []

with torch.no_grad():
    pbar = tqdm(test_dataloader)
    for i, data in enumerate(pbar):
        if args.dataset == 'kitti':
            input_h, input_w = data['features'][0].shape[2] * 32, data['features'][0].shape[3] * 32
            for j in range(len(data['features'])):
                data['features'][j] = data['features'][j].cuda()
            targets = data['flow_2d']
            pred_flow = model(data['features'])

            pred_flow = pred_flow.clamp(-2.6,2.6).detach()
            pred_flow = pred_flow[:input_h, :input_w]
            total_mse += get_flow_loss(pred_flow, data['flow_2d'].cuda(), type='mse').item()
            total_epe += EPE(resize_flow2d(pred_flow, targets.shape[2], targets.shape[3]), targets[:, :2].cuda()).item()
            # print(i, total_mse, total_epe)
            pred_flow = pred_flow.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
            save_flow_png(os.path.join(args.save_path, '%06d_10.png' % i), pred_flow, scale=12800.0)
        else:
                  
            inputs, targets = data['features'], data['flow'].cuda()

            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda()
            
            outputs = model(inputs)

            loss_val = realEPE(outputs,targets)
            average_EPE.append(loss_val)
          
    
if args.dataset == 'kitti':
    total_mse = total_mse/len(test_dataloader)
    total_epe = total_epe/len(test_dataloader)

    print(f'epe_loss: {total_epe}, mse_loss: {total_mse}')

else:

    print(f"Average EPE: {sum(average_EPE)/len(average_EPE)}")