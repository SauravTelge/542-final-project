import os
import torch
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.utils.data as data



# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def make_dataset(dataset_dir, dataset_type="clean"):
    flow_dir = "flow"
    assert os.path.isdir(os.path.join(dataset_dir, flow_dir))

    img_dir = dataset_type
    assert os.path.isdir(os.path.join(dataset_dir, img_dir))

    features = []
    flow_list = []

    for flow_map in sorted(
        glob.glob(os.path.join(dataset_dir, flow_dir, "*", "*.flo"))
    ):
        flow_map = os.path.relpath(flow_map, os.path.join(dataset_dir, flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        # if not(scene_dir == 'alley_1' or scene_dir == 'alley_2'): continue
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split("_")
        frame_nb = int(frame_nb)

        featureDir = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        img1_features = [ os.path.join(
            img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb)
        ) for feat in featureDir ]

        img2_features = [ os.path.join(
            img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb+1)
        ) for feat in featureDir ]

        flow_map = os.path.join(flow_dir, flow_map)
        if not (
            os.path.isfile(os.path.join(dataset_dir, img1_features[0]))
            and os.path.isfile(os.path.join(dataset_dir, img2_features[0]))
        ):
            continue

        features.append([img1_features, img2_features])
        flow_list.append(flow_map)

    
    X_train, X_test, y_train, y_test= train_test_split(features, flow_list, test_size=0.2, shuffle=True)
    return X_train, X_test, y_train, y_test

def make_dataset_eval(dataset_dir, dataset_type="clean"):
    flow_dir = "flow"
    assert os.path.isdir(os.path.join(dataset_dir, flow_dir))

    img_dir = dataset_type
    assert os.path.isdir(os.path.join(dataset_dir, img_dir))

    features = []
    flow_list = []

    for flow_map in sorted(
        glob.glob(os.path.join(dataset_dir, flow_dir, "*", "*.flo"))
    ):
        flow_map = os.path.relpath(flow_map, os.path.join(dataset_dir, flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        # if not(scene_dir == 'alley_1' or scene_dir == 'alley_2'): continue
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split("_")
        frame_nb = int(frame_nb)

        featureDir = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        img1_features = [ os.path.join(
            img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb)
        ) for feat in featureDir ]

        img2_features = [ os.path.join(
            img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb+1)
        ) for feat in featureDir ]

        flow_map = os.path.join(flow_dir, flow_map)
        if not (
            os.path.isfile(os.path.join(dataset_dir, img1_features[0]))
            and os.path.isfile(os.path.join(dataset_dir, img2_features[0]))
        ):
            continue

        features.append([img1_features, img2_features])
        flow_list.append(flow_map)

    return features, flow_list

def load_flo(path):
    with open(path, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert 202021.25 == magic, "Magic number incorrect. Invalid .flo file"
        # h = np.fromfile(f, np.int32, count=1)[0]
        # w = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        # print(f"Inside load_flow: h={h} and w={w}")
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    # data2D = np.resize(data, (2, h, w))
    data2D = np.resize(data, (h, w, 2))
    # data2D = np.resize(data, (2, w, h))
    # print(f"shape of resized data: {data2D.shape}")
    return data2D



def default_loader(root, path_img1_feat, path_img2_feat, path_flo):
    feat_1 = [ torch.from_numpy(np.load(os.path.join(root, path))) for path in path_img1_feat]
    feat_2 = [ torch.from_numpy(np.load(os.path.join(root, path))) for path in path_img2_feat]

    features = [ torch.cat((feat_1[index],feat_2[index]), dim=0) for index in range(4)]
    flo = os.path.join(root, path_flo)
    return features, load_flo(flo)


class ListDataset(data.Dataset):
    def __init__(
        self,
        root,
        feature_list,
        flow_list,
        loader=default_loader,
        target_transform=None
    ):

        self.root = root
        self.feature_list = feature_list
        self.flow_list = flow_list
        self.loader = loader

        self.target_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[-5.1417844564964374, 6.317663301890716], std=[11.778960670232772, 8.836427354713281]),
        ]
        )

    def __getitem__(self, index):
        img1_features_paths, img2_features_paths = self.feature_list[index]
        target_path = self.flow_list[index]

        img_features, target = self.loader(self.root, img1_features_paths, img2_features_paths, target_path)
        data_dict = {'index': index}
        data_dict['features'] = img_features    

        # if self.target_transform is not None:
            # print(f"Target shape in data loader: {target.shape}")
        # print(target[:10])
        target = self.target_transform(target)
        # print(target[:10])
        # print(f"Shape of target in data loader: {target.shape}")
        # print(f"Range of target: mean: {target.mean()} std: {target.std()}")
        data_dict['flow'] = target
        
        return data_dict

    def __len__(self):
        return len(self.flow_list)
