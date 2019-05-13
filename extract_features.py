import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('-mode', default='rgb', type=str, help='rgb or flow')
parser.add_argument('-load_model', default='./models/rgb_imagenet.pt', type=str)
parser.add_argument('-root', default='../data/BL_and_PL', type=str)
parser.add_argument('-gpu', default='0', type=str)
parser.add_argument('-save_dir', default='./', type=str)
parser.add_argument('-temporal_window',default=5,type=int)


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms


import numpy as np

from pytorch_i3d import InceptionI3d

# from charades_dataset_full import Charades as Dataset
from mydataset import myDataset as Dataset


def run(max_steps=64e3, mode='rgb', root='../data/BL_and_PL', split='../data/BL_and_PL/new_annotations.json', batch_size=1, load_model='', save_dir='args.save_dir'):
    # setup dataset
    
    #test_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                      videotransforms.RandomHorizontalFlip(),
    #])

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(split, 'training', root, mode, test_transforms, num=-1, save_dir=save_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=0, pin_memory=True)
    dataloaders = {'train': dataloader}
    datasets = {'train': dataset}
    
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2, temp_window = args.temporal_window)
        input_padder = nn.ReplicationPad3d(padding=(0,0,0,0,
                                    args.temporal_window//2,args.temporal_window//2))
    #, final_endpoint= 'Mixed_5c'
    else:
        i3d = InceptionI3d(400,in_channels=3, temp_window = args.temporal_window) 
        input_padder = nn.ReplicationPad3d(padding=(0,0,0,0,
                                    args.temporal_window//2,args.temporal_window//2))

    i3d.load_state_dict(torch.load(load_model))
    i3d.cuda()

    for phase in ['train']:#, 'val']:
        i3d.train(False)  # Set model to evaluate mode
                
        tot_loss = 0.0
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
                    
        # Iterate over data.
        for data in dataloaders[phase]:
            # get the inputs
            inputs, labels, name = data

            print('extracting {} features for {} and tw {}'.format(mode, name[0],args.temporal_window))

            #if os.path.exists(os.path.join(save_dir, name[0]+'.npy')):
            #    continue

            b,c,t,h,w = inputs.shape
            if t > 1600:
                features = []
                for start in range(1, t-56, 1600):
                    end = min(t-1, start+1600+56)
                    start = max(1, start-48)

                    ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)

                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                np.save(os.path.join(save_dir, name[0]), np.concatenate(features, axis=0))
            else:
                # Temporally pad inputs such that output temporal dimension conserved:
                no_frames = inputs.shape[2]
                inputs = input_padder(inputs)

                per_frame_features = []#torch.zeros((1,1024,1,1,1))

                # We want per-frame features. Authors of MS-TCN slid temporal window over 
                # each frame and input that to the network.           
             
                for w in range(no_frames):

                    windowed_inputs = inputs[:,:, w:(w+(args.temporal_window)), :,:].cuda()
                    features = i3d.extract_features(windowed_inputs)

                    per_frame_features.append(features.cpu().data)
                    if w % 10 == 0:
                        print('         {}'.format(w) )

                np.save(os.path.join(save_dir, name[0]), 
                        np.concatenate(per_frame_features,axis=2)[0,:,:,0,0])

if __name__ == '__main__':
    # need to add argparse
    run(mode=args.mode, root=args.root, load_model=args.load_model, save_dir=args.save_dir)
