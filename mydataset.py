import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path
import pdb

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []

  vid_dir = os.path.join(image_dir, 'ims', vid)


  vid_files = [d.name for d in os.scandir(vid_dir) if d.is_file()]
  vid_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
  # lsorted = sorted(vid_files,key=lambda x: int(os.path.splitext(x)[0]))


  for i in range(start, start+num):
    img = cv2.imread(os.path.join(vid_dir, vid_files[i] ))[:, :, [2, 1, 0]]# (BGR -> RGB)
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def make_dataset(split_file, split, root, mode, num_classes=10):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    train_dir = os.path.join(root, 'ims')
    videos_dir = [d.name for d in os.scandir(train_dir) if d.is_dir()]
    video_dir_idx = [int(d[d.rfind('p') + 1 : d.rfind('.')].replace('_','')) for d in videos_dir] 
    video_dir_idx = np.array(video_dir_idx).argsort()
    videos_dir = list(np.array(videos_dir)[video_dir_idx])
   
    class_list = []
    print('Number of annotations: {}'.format(len(data['annotations'])))

    for class_i in range(len(data['annotations'])):
        activity = data['annotations'][class_i]['attributes']['activity']

        # if activity == '__undefined__':
        #     print('id: {}'.format(data['annotations'][class_i]['id']))
        #     print('image_id: {}'.format(data['annotations'][class_i]['image_id']))
            # pdb.set_trace()

        if activity not in class_list:
            class_list.append(activity)

    class_to_id = {class_i:idx for idx, class_i in enumerate(class_list, 1) if class_i != '__undefined__'}
    class_to_id['__undefined__']=0

    video_frame_idx = 0

    for video_i in videos_dir:
        # if data[vid]['subset'] != split:
        #     continue
        
        video_i_path = os.path.join(train_dir, video_i)

        num_frames = len([1 for file in os.scandir(video_i_path) if file.is_file()])

        if mode == 'flow':
            num_frames = num_frames//2
            
        if num_frames < 66:
            video_frame_idx += num_frames
            continue

        label = np.zeros((num_classes,num_frames), np.float32)

        
        for fr_idx in range(0,num_frames,1):
            fr_class = data['annotations'][video_frame_idx]['attributes']['activity']
            label[class_to_id.get(fr_class,0), fr_idx] = 1 # binary classification
            video_frame_idx += 1

        dataset.append((video_i, label, num_frames))

    return dataset


class myDataset(data_utl.Dataset):

    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        start_f = random.randint(1,nf-65)

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = label[:, start_f:start_f+64]

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)



# if __name__ == "__main__":
#     # execute only if run as a script
#     train_split = '../data/BrickLaying/annotations.json'
#     root = '../data/BrickLaying'
