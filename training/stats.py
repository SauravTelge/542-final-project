import os
import torch
import glob
import numpy as np

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

def make_dataset(dataset_dir, dataset_type="clean"):
    flow_dir = "flow"
    assert os.path.isdir(os.path.join(dataset_dir, flow_dir))

    # img_dir = dataset_type
    # assert os.path.isdir(os.path.join(dataset_dir, img_dir))

    # features = []
    flow_list = []
    mean_0 = []
    mean_1 = []
    std_0 = []
    std_1 = []
    for flow_map in sorted(
        glob.glob(os.path.join(dataset_dir, flow_dir, "*", "*.flo"))
    ):
        flow_map = os.path.relpath(flow_map, os.path.join(dataset_dir, flow_dir))

        # scene_dir, filename = os.path.split(flow_map)
        # # if not(scene_dir == 'alley_1' or scene_dir == 'alley_2'): continue
        # no_ext_filename = os.path.splitext(filename)[0]
        # prefix, frame_nb = no_ext_filename.split("_")
        # frame_nb = int(frame_nb)

        # featureDir = ['feature_0', 'feature_1', 'feature_2', 'feature_3']
        # img1_features = [ os.path.join(
        #     img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb)
        # ) for feat in featureDir ]

        # img2_features = [ os.path.join(
        #     img_dir, feat , scene_dir, "{}_{:04d}.npy".format(prefix, frame_nb+1)
        # ) for feat in featureDir ]

        flow_map = os.path.join(flow_dir, flow_map)
        # if not (
        #     os.path.isfile(os.path.join(dataset_dir, img1_features[0]))
        #     and os.path.isfile(os.path.join(dataset_dir, img2_features[0]))
        # ):
            # continue

        # features.append([img1_features, img2_features])
        # flow_list.append(flow_map)
        flow_map = os.path.join(dataset_dir, flow_map)
        flow_data = load_flo(flow_map)
        mean_0.append(flow_data[:,:,0].mean())
        mean_1.append(flow_data[:,:,1].mean())
        std_0.append(flow_data[:,:,0].std())
        std_1.append(flow_data[:,:,1].std())
        # print(flow_map)

    
    # X_train, X_test, y_train, y_test= train_test_split(features, flow_list, test_size=0.2, shuffle=True)
    return [sum(mean_0)/len(mean_0),sum(mean_1)/len(mean_1)], [sum(std_0)/len(std_0),sum(std_1)/len(std_1)] 

dataset_dir = '/home/rutwik/542project/dift/project_sintel_files/Sintel'
mean_list, std_list = make_dataset(dataset_dir)

print(mean_list)
print(std_list)