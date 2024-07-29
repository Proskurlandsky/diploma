import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import torch
from mean_std import norm
import csv
import random

class ben(Dataset):
    def __init__(self, root, train):
        self.train = train
        if train:
            with open('train.csv') as csvfile:
                self.folders = list(csv.reader(csvfile))
                
        if not train:
            with open('test.csv') as csvfile:
                self.folders = list(csv.reader(csvfile))
            
        with open('label_indices.json', 'rb') as f:
            self.label_indices = json.load(f)

        self.root = root
            
    def __getitem__(self, index):
        patch = []
        bands = ['01', '02', '03', '04', '05', '06', '07', '08', '8A', '09', '11', '12']
        for band in bands:
            patch.append(os.path.join(self.root, self.folders[index][0], 
                                      self.folders[index][0] + '_B' + band + '.tif'))
        im = [Image.open(path) for path in patch]
        im = [np.asarray(img, dtype='float32') for img in im]
        for i in range(12):
            im[i] = (im[i] - norm['mean'][i]) / norm['std'][i]
        im1 = np.asarray([im[1], im[2], im[3], im[7]])
        im1 = torch.FloatTensor(im1)
        im2 = np.asarray([im[4], im[5], im[6], im[8], im[10], im[11]])
        im2 = torch.FloatTensor(im2)
        im3 = np.asarray([im[0], im[9]])
        im3 = torch.FloatTensor(im3)
        '''if self.train:
            p = random.choices(range(3), weights=[.3, .3, .4])
            if p[0] == 0:
                im1 = torch.zeros_like(im1)
            if p[0] == 1:
                im2 = torch.zeros_like(im2)'''
        '''if not self.train:
            im1 = torch.zeros_like(im1)
            im2 = torch.zeros_like(im2)
            im3 = torch.zeros_like(im3)'''
        label_path = os.path.join(self.root, self.folders[index][0], self.folders[index][0] + '_labels_metadata.json')
        with open(label_path, 'rb') as f:
            patch_json = json.load(f)
        original_labels = patch_json['labels']
        lbl = torch.zeros(len(self.label_indices['original_labels'].keys()))
        for label in original_labels:
            lbl[self.label_indices['original_labels'][label]] = 1
        
        return im1, im2, im3, lbl
    
    def __len__(self):
        return len(self.folders)
    
if __name__ == '__main__':
    root = '/data/Tselkovoy/dataset/BigEarthNet-v1.0'
    train_data = ben(root = root, variant = 'test')
    data_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    print(len(data_loader))
    x1, x2, x3, train_labels = next(iter(data_loader))
    print(train_labels)