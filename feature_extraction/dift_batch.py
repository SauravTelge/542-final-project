import argparse
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
import time
import os
import numpy as np

def main(args):
    start_time = time.time()
    dift = SDFeaturizer(args.model_id)

    # listing all subfolders
    folderList = os.listdir(args.input_path)

    # defining names of folders of features
    featureIndex = ['feature_0','feature_1', 'feature_2', 'feature_3']
    noGoFolders = ['alley_1', 'alley_2', 'market_6', 'cave_4', 'ambush_2', 'ambush_4', 'ambush_5','shaman_2', 'bandage_1',
                   'sleeping_1', 'bamboo_2', 'market_5', 'ambush_7', 'cave_2', 'bamboo_1', 'temple_2', 'temple_3', 'ambush_6',
                   'bandage_2', 'market_2', 'shaman_3']

    featurePaths = []
    # making folders for features
    for featureName in featureIndex:
        featurePaths.append(args.input_path+featureName+'/')
        if not os.path.exists(args.input_path+featureName): os.makedirs(args.input_path+featureName)
    

    # going in each sub folder and starting feature extraction
    for folder in folderList:
        if folder in featureIndex: continue
        if folder in noGoFolders: continue
        print(f"\nIn folder: {folder} \n")
        
        featurePathsForFolder = []
        for featurePath in featurePaths:
            featurePathsForFolder.append(featurePath+folder+'/')
            if not os.path.exists(featurePath+folder): os.makedirs(featurePath+folder)

        # listing all files in a sub folder
        fileList = os.listdir(args.input_path+folder)
        
        i = 1
        print(f"Total files: {len(fileList)}\n")
        for imageFile in fileList:
            # print(f"Working on file: {imageFile}")
            
            if not(imageFile.endswith('.jpg') or imageFile.endswith('.png')): continue

            frameName = os.path.splitext(imageFile)[0]

            # if i>0: break

            img = Image.open(args.input_path+folder+'/'+imageFile).convert('RGB')
            if args.img_size[0] > 0:
                img = img.resize(args.img_size)
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2

            unetFeatureIndexes = [0,1,2,3]

            for unetFeatureIndex in unetFeatureIndexes:
                # exampleTensor = torch.empty(1280,14,32, dtype=torch.float32)
                ft = dift.forward(img_tensor,
                                  prompt=args.prompt,
                                  t=args.t,
                                  up_ft_index=unetFeatureIndex,
                                  ensemble_size=args.ensemble_size)
                np.save(featurePathsForFolder[unetFeatureIndex]+frameName+'.npy', ft.squeeze(0).cpu())
            print(f"Done:{i}/{len(fileList)}")
            i+=1
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='''extract dift from input image, and save it as torch tenosr,
                    in the shape of [c, h, w].''')
    
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into diffusion model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1', type=str, 
                        help='model_id of the diffusion model in huggingface')
    parser.add_argument('--t', default=261, type=int, 
                        help='time step for diffusion, choose from range [0, 1000]')
    parser.add_argument('--up_ft_index', default=1, type=int, choices=[0, 1, 2 ,3],
                        help='which upsampling block of U-Net to extract the feature map')
    parser.add_argument('--prompt', default='', type=str,
                        help='prompt used in the stable diffusion')
    parser.add_argument('--ensemble_size', default=8, type=int, 
                        help='number of repeated images in each batch used to get features')
    parser.add_argument('--input_path', type=str,
                        help='path to the input image folder')
    parser.add_argument('--output_path', type=str, default='dift.pt',
                        help='path to save the output features as torch tensor')
    args = parser.parse_args()
    main(args)