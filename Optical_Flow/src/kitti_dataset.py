import os
import cv2
import numpy as np
import torch.utils.data
from utils import load_flow_png, zero_padding


class KITTI(torch.utils.data.Dataset):
    def __init__(self, root_dir, split='training'):
        assert os.path.isdir(root_dir)
        
        if 'training' in split:
            self.root_dir = os.path.join(root_dir, 'training')
        else:
            self.root_dir = os.path.join(root_dir, 'testing')

        self.split = split
        self.indices = np.arange(200)
       
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
       
        np.random.seed(23333)

        index = self.indices[i]
        
        data_dict = {'index': index}
        # print('index', index)
        img1_feats_0_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_0', '%06d_10.pt' % index))
        img2_feats_0_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_0', '%06d_11.pt' % index))
        feats_0_0 = torch.cat((img1_feats_0_0, img2_feats_0_0), dim=0)

        img1_feats_0_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_41', '%06d_10.pt' % index))
        img2_feats_0_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_41', '%06d_11.pt' % index))
        feats_0_41 = torch.cat((img1_feats_0_41, img2_feats_0_41), dim=0)

        img1_feats_0_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_261', '%06d_10.pt' % index))
        img2_feats_0_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_261', '%06d_11.pt' % index))
        feats_0_261 = torch.cat((img1_feats_0_261, img2_feats_0_261), dim=0)

        feats_0 = (0.7 * feats_0_0) + (0.2 * feats_0_41) + (0.1 * feats_0_261)

        # img1_feats_0 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_0', '%06d_10.npy' % index)))
        # img2_feats_0 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_0', '%06d_11.npy' % index)))
        # feats_0 = torch.cat((img1_feats_0, img2_feats_0), dim=0)
        # feats_0 = torch.randn((2560, 12, 39))

        img1_feats_1_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_0', '%06d_10.pt' % index))
        img2_feats_1_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_0', '%06d_11.pt' % index))
        feats_1_0 = torch.cat((img1_feats_1_0, img2_feats_1_0), dim=0)

        img1_feats_1_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_41', '%06d_10.pt' % index))
        img2_feats_1_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_41', '%06d_11.pt' % index))
        feats_1_41 = torch.cat((img1_feats_1_41, img2_feats_1_41), dim=0)

        img1_feats_1_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_261', '%06d_10.pt' % index))
        img2_feats_1_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_261', '%06d_11.pt' % index))
        feats_1_261 = torch.cat((img1_feats_1_261, img2_feats_1_261), dim=0)

        feats_1 = (0.7 * feats_1_0) + (0.2 * feats_1_41) + (0.1 * feats_1_261)
       
        # img1_feats_1 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_1', '%06d_10.npy' % index)))
        # img2_feats_1 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_1', '%06d_11.npy' % index)))
        # feats_1 = torch.cat((img1_feats_1, img2_feats_1), dim=0)
        # feats_1 = torch.randn((2560, 23, 78))

        img1_feats_2_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_0', '%06d_10.pt' % index))
        img2_feats_2_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_0', '%06d_11.pt' % index))
        feats_2_0 = torch.cat((img1_feats_2_0, img2_feats_2_0), dim=0)

        img1_feats_2_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_41', '%06d_10.pt' % index))
        img2_feats_2_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_41', '%06d_11.pt' % index))
        feats_2_41 = torch.cat((img1_feats_2_41, img2_feats_2_41), dim=0)

        img1_feats_2_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_261', '%06d_10.pt' % index))
        img2_feats_2_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_261', '%06d_11.pt' % index))
        feats_2_261 = torch.cat((img1_feats_2_261, img2_feats_2_261), dim=0)

        feats_2 = (0.7 * feats_2_0) + (0.2 * feats_2_41) + (0.1 * feats_2_261)
        
        # img1_feats_2 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_2', '%06d_10.npy' % index)))
        # img2_feats_2 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_2', '%06d_11.npy' % index)))
        # feats_2 = torch.cat((img1_feats_2, img2_feats_2), dim=0)
        # feats_2 = torch.randn((1280, 46, 155))

        img1_feats_3_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_0', '%06d_10.pt' % index))
        img2_feats_3_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_0', '%06d_11.pt' % index))
        feats_3_0 = torch.cat((img1_feats_3_0, img2_feats_3_0), dim=0) 

        img1_feats_3_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_41', '%06d_10.pt' % index))
        img2_feats_3_41 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_41', '%06d_11.pt' % index))
        feats_3_41 = torch.cat((img1_feats_3_41, img2_feats_3_41), dim=0) 

        img1_feats_3_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_261', '%06d_10.pt' % index))
        img2_feats_3_261 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_261', '%06d_11.pt' % index))
        feats_3_261 = torch.cat((img1_feats_3_261, img2_feats_3_261), dim=0) 

        feats_3 = (0.7 * feats_3_0) + (0.2 * feats_3_41) + (0.1 * feats_3_261)

        # img1_feats_3 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_3', '%06d_10.npy' % index)))
        # img2_feats_3 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_3', '%06d_11.npy' % index)))
        # feats_3 = torch.cat((img1_feats_3, img2_feats_3), dim=0)   
        # feats_3 = torch.randn((640, 46, 155))  

        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))

        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1)
       
        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        # padding_h, padding_w = 376, 1242
        
        # flow_2d = zero_padding(flow_2d, padding_h, padding_w)
    
        feats = [feats_0, feats_1, feats_2, feats_3]

        data_dict['features'] = feats
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        

        return data_dict

class KITTI_Test(torch.utils.data.Dataset):
    def __init__(self, root_dir, ts, split='training'):
        assert os.path.isdir(root_dir)
        
        if 'training' in split:
            self.root_dir = os.path.join(root_dir, 'training')
        else:
            self.root_dir = os.path.join(root_dir, 'testing')

        self.split = split
        self.ts = ts
        self.indices = np.arange(200)
       
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
       
        np.random.seed(23333)

        index = self.indices[i]
        
        data_dict = {'index': index}
        # print('index', index)
        img1_feats_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_' + str(self.ts) + '/' + '%06d_10.pt' % index))
        img2_feats_0 = torch.load(os.path.join(self.root_dir, 'image_2/feats_0_' + str(self.ts) + '/' + '%06d_11.pt' % index))
        feats_0 = torch.cat((img1_feats_0, img2_feats_0), dim=0)

        # img1_feats_0 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_0', '%06d_10.npy' % index)))
        # img2_feats_0 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_0', '%06d_11.npy' % index)))
        # feats_0 = torch.cat((img1_feats_0, img2_feats_0), dim=0)
        # feats_0 = torch.randn((2560, 12, 39))

        img1_feats_1 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_' + str(self.ts) + '/' + '%06d_10.pt' % index))
        img2_feats_1 = torch.load(os.path.join(self.root_dir, 'image_2/feats_1_' + str(self.ts) + '/' + '%06d_11.pt' % index))
        feats_1 = torch.cat((img1_feats_1, img2_feats_1), dim=0)
       
        # img1_feats_1 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_1', '%06d_10.npy' % index)))
        # img2_feats_1 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_1', '%06d_11.npy' % index)))
        # feats_1 = torch.cat((img1_feats_1, img2_feats_1), dim=0)
        # feats_1 = torch.randn((2560, 23, 78))

        img1_feats_2 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_' + str(self.ts) + '/' + '%06d_10.pt' % index))
        img2_feats_2 = torch.load(os.path.join(self.root_dir, 'image_2/feats_2_' + str(self.ts) + '/' + '%06d_11.pt' % index))
        feats_2 = torch.cat((img1_feats_2, img2_feats_2), dim=0)
        
        # img1_feats_2 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_2', '%06d_10.npy' % index)))
        # img2_feats_2 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_2', '%06d_11.npy' % index)))
        # feats_2 = torch.cat((img1_feats_2, img2_feats_2), dim=0)
        # feats_2 = torch.randn((1280, 46, 155))

        img1_feats_3 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_' + str(self.ts) + '/' + '%06d_10.pt' % index))
        img2_feats_3 = torch.load(os.path.join(self.root_dir, 'image_2/feats_3_' + str(self.ts) + '/' + '%06d_11.pt' % index))
        feats_3 = torch.cat((img1_feats_3, img2_feats_3), dim=0) 

        # img1_feats_3 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_3', '%06d_10.npy' % index)))
        # img2_feats_3 = torch.from_numpy(np.load(os.path.join(self.root_dir, 'image_2/feats_3', '%06d_11.npy' % index)))
        # feats_3 = torch.cat((img1_feats_3, img2_feats_3), dim=0)   
        # feats_3 = torch.randn((640, 46, 155))  

        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))

        flow_2d = np.concatenate([flow_2d, flow_2d_mask[..., None].astype(np.float32)], axis=-1)
       
        # images from KITTI have various sizes, padding them to a unified size of 1242x376
        # padding_h, padding_w = 376, 1242
        
        # flow_2d = zero_padding(flow_2d, padding_h, padding_w)
    
        feats = [feats_0, feats_1, feats_2, feats_3]

        data_dict['features'] = feats
        data_dict['flow_2d'] = flow_2d.transpose([2, 0, 1])
        

        return data_dict
    



