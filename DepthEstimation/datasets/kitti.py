import os
import numpy as np
import PIL.Image as Image
import torch
import torch.utils.data as data


def crop_img(img, left, top, crph, crpw):
    if img.ndim == 2:
        img_cropped = img[top:top + crph, left:left + crpw]
    elif img.ndim == 3:
        img_cropped = img[top:top + crph, left:left + crpw, :]
    return img_cropped

class KITTI_Train(data.Dataset):
    def __init__(self, img_root, anno_root, split='train'):
        super(KITTI_Train, self).__init__()
        self.img_root = img_root
        self.split = split
        self.anno_root = anno_root

        self.folder_paths = [os.path.join(self.img_root, file) for file in sorted(os.listdir(self.img_root))]
        self.folders = []

        for i in range(len(self.folder_paths)):
            # print(sorted(os.listdir(self.folder_paths[i]))[0])
            self.folders.append(sorted(os.listdir(self.folder_paths[i]))[0])
        
        self.img_paths = []
        self.depth_paths = []

        for i in range(len(self.folder_paths)):
            gt_path = os.path.join(self.anno_root, self.split + "/" + self.folders[i] + "/proj_depth/groundtruth/image_02")
            gt_paths = [os.path.join(gt_path, file) for file in sorted(os.listdir(gt_path))]
            self.depth_paths = self.depth_paths + gt_paths
            img_path = os.path.join(self.folder_paths[i], self.folders[i] + "/image_02/data")
            imgPaths = [os.path.join(img_path, file) for file in sorted(os.listdir(gt_path))]
            self.img_paths = self.img_paths + imgPaths

        
    def read_rgb(self, file_path):
        return np.array(Image.open(file_path)).astype(np.uint8)

    def read_depth(self, file_path):
        return np.array(Image.open(file_path)).astype(np.float32) / 256.0

    def __getitem__(self, idx):
       
        file_name = os.path.basename(self.img_paths[idx]).split('.')[0]
        folder_path = os.path.dirname(self.img_paths[idx])

        feats_0_0 = torch.load(os.path.join(folder_path, 'feats_0_0/' + file_name + '.pt'))
        feats_1_0 = torch.load(os.path.join(folder_path, 'feats_1_0/' + file_name + '.pt'))
        feats_2_0 = torch.load(os.path.join(folder_path, 'feats_2_0/' + file_name + '.pt'))
        feats_3_0 = torch.load(os.path.join(folder_path, 'feats_3_0/' + file_name + '.pt'))

        feats_0_41 = torch.load(os.path.join(folder_path, 'feats_0_41/' + file_name + '.pt'))
        feats_1_41 = torch.load(os.path.join(folder_path, 'feats_1_41/' + file_name + '.pt'))
        feats_2_41 = torch.load(os.path.join(folder_path, 'feats_2_41/' + file_name + '.pt'))
        feats_3_41 = torch.load(os.path.join(folder_path, 'feats_3_41/' + file_name + '.pt'))

        feats_0_261 = torch.load(os.path.join(folder_path, 'feats_0_261/' + file_name + '.pt'))
        feats_1_261 = torch.load(os.path.join(folder_path, 'feats_1_261/' + file_name + '.pt'))
        feats_2_261 = torch.load(os.path.join(folder_path, 'feats_2_261/' + file_name + '.pt'))
        feats_3_261 = torch.load(os.path.join(folder_path, 'feats_3_261/' + file_name + '.pt'))

        feats_0 = (0.7 * feats_0_0) + (0.2 * feats_0_41) + (0.1 * feats_0_261)
        feats_1 = (0.7 * feats_1_0) + (0.2 * feats_1_41) + (0.1 * feats_1_261)
        feats_2 = (0.7 * feats_2_0) + (0.2 * feats_2_41) + (0.1 * feats_2_261)
        feats_3 = (0.7 * feats_3_0) + (0.2 * feats_3_41) + (0.1 * feats_3_261)

        feats = [feats_0, feats_1, feats_2, feats_3]

        
        depth = self.read_depth(self.depth_paths[idx])

       
        sample = {
            'features' : feats,
            'depth': torch.from_numpy(depth).unsqueeze(0).float(),
            'index': file_name
        }

        return sample
   
    def __len__(self):
        return len(self.img_paths)
    
    
class KITTI_Test(data.Dataset):
    def __init__(self, img_root, anno_root, ts, split='train'):
        super(KITTI_Test, self).__init__()
        self.img_root = img_root
        self.split = split
        self.anno_root = anno_root
        self.ts = ts

        self.folder_paths = [os.path.join(self.img_root, file) for file in sorted(os.listdir(self.img_root))]
        self.folders = []

        for i in range(len(self.folder_paths)):
            # print(sorted(os.listdir(self.folder_paths[i]))[0])
            self.folders.append(sorted(os.listdir(self.folder_paths[i]))[0])
        
        self.img_paths = []
        self.depth_paths = []

        for i in range(len(self.folder_paths)):
            gt_path = os.path.join(self.anno_root, self.split + "/" + self.folders[i] + "/proj_depth/groundtruth/image_02")
            gt_paths = [os.path.join(gt_path, file) for file in sorted(os.listdir(gt_path))]
            self.depth_paths = self.depth_paths + gt_paths
            img_path = os.path.join(self.folder_paths[i], self.folders[i] + "/image_02/data")
            imgPaths = [os.path.join(img_path, file) for file in sorted(os.listdir(gt_path))]
            self.img_paths = self.img_paths + imgPaths

        
    def read_rgb(self, file_path):
        return np.array(Image.open(file_path)).astype(np.uint8)

    def read_depth(self, file_path):
        return np.array(Image.open(file_path)).astype(np.float32) / 256.0

    def __getitem__(self, idx):
       
        file_name = os.path.basename(self.img_paths[idx]).split('.')[0]
        folder_path = os.path.dirname(self.img_paths[idx])

        feats_0 = torch.load(os.path.join(folder_path, 'feats_0_' + str(self.ts) + '/' + file_name + '.pt'))
        feats_1 = torch.load(os.path.join(folder_path, 'feats_1_' + str(self.ts) + '/' + file_name + '.pt'))
        feats_2 = torch.load(os.path.join(folder_path, 'feats_2_' + str(self.ts) + '/' + file_name + '.pt'))
        feats_3 = torch.load(os.path.join(folder_path, 'feats_3_' + str(self.ts) + '/' + file_name + '.pt'))

        feats = [feats_0, feats_1, feats_2, feats_3]

        
        depth = self.read_depth(self.depth_paths[idx])

       
        sample = {
            'features' : feats,
            'depth': torch.from_numpy(depth).unsqueeze(0).float(),
            'index': file_name
        }

        return sample
   
    def __len__(self):
        return len(self.img_paths)
    
    