import json
import os
import torch

with open('label_indices.json', 'rb') as g:
    label_indices = json.load(g)

root = '/data/Tselkovoy/dataset/BigEarthNet-v1.0'

data = {}

folders = os.listdir(root)

print('listdir is made')

i = 0

for folder in folders:
    label_path = os.path.join(root, folder, folder + '_labels_metadata.json')
    with open(label_path, 'rb') as f:
        patch_json = json.load(f)
    original_labels = patch_json['labels']
    lbl = [0]*43
    for label in original_labels:
        lbl[label_indices['original_labels'][label]] = 1
    data[i] = lbl
    i+=1
    print(i)
    
with open('labels.json', 'w') as outfile:
    json.dump(data, outfile)

