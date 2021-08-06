import imageio
from PIL import Image
import os
from torch.utils import data
import numpy as np

class SegDataset(object):
    def __init__(self, root, split, transform): 

        self.root = root
        self.split = split
        self.transform = transform

        listfile = os.path.join(self.root, self.split)
        with open(listfile, 'r') as f:
            lines = f.readlines()
            self.img_ids = [line.strip() for line in lines if line.strip() != ""]

        self.files = []
        for item in self.img_ids:
            image_path, label_path = item.split()
            #name = os.path.splitext(os.path.basename(label_path))[0]
            name = os.path.splitext(os.path.basename(image_path))[0]
            img_file = os.path.join(self.root, image_path)
            label_file = os.path.join(self.root, label_path)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

        print('Split: %s with %d files.' % (listfile, len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        img_path, label_path, name = datafiles['img'], datafiles['label'], datafiles['name']
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return {'Img': image, 'Label': label, 'Name': name}

class Cityscapes(SegDataset):
    pass

class GTAV(SegDataset):
    pass

class SYNTHIA(SegDataset):
    def __getitem__(self, index):
        datafiles = self.files[index]
        img_path, label_path, name = datafiles['img'], datafiles['label'], datafiles['name']
        image = Image.open(img_path).convert('RGB')
        label = Image.fromarray(np.uint8(imageio.imread(label_path, 'PNG-FI')[:, :, 0]))

        if self.transform is not None:
            image, label = self.transform(image, label)

        return {'Img': image, 'Label': label, 'Name': name}

class SegDualDataset(object):
    def __init__(self, root_S, split_S, root_T, split_T, transform): 

        self.root_S = root_S
        self.split_S = split_S
        self.root_T = root_T
        self.split_T = split_T

        listfile_S = os.path.join(self.root_S, self.split_S)
        self.files_S = self.construct_filelist(listfile_S)
        print('Source split: %s with %d files.' % (listfile_S, len(self.files_S)))

        listfile_T = os.path.join(self.root_T, self.split_T)
        self.files_T = self.construct_filelist(listfile_T, False)
        print('Target split: %s with %d files.' % (listfile_T, len(self.files_T)))

        self.len_S = len(self.files_S)
        self.len_T = len(self.files_T)
        self.max_len = max(self.len_S, self.len_T)

        self.transform = transform

    def construct_filelist(self, listfile, with_label=True):
        with open(listfile, 'r') as f:
            lines = f.readlines()
            img_ids = [line.strip() for line in lines if line.strip() != ""]

        files = []
        for item in img_ids:
            image_path, label_path = item.split('\t')
            #name = os.path.splitext(os.path.basename(label_path))[0]
            name = os.path.splitext(os.path.basename(image_path))[0]
            img_file = os.path.join(self.root, image_path)
            if with_label:
                label_file = os.path.join(self.root, label_path)
                files.append({
                    "img": img_file,
                    "label": label_file,
                    "name": name
                })
            else:
                files.append({
                    "img": img_file,
                    "name": name
                })

        return files

    def __len__(self):
        return self.max_len 

    def decode_data(self, files, index):
        ind = index % len(files)
        datafiles = files[ind]
        img_path, name = datafiles['img'], datafiles['name']
        image = Image.open(img_path).convert('RGB')
        label = None
        if 'label' in datafiles:
            label_path = datafiles['label']
            label = Image.open(label_path)
        return image, label, name
         
    def __getitem__(self, index):
        image_S, label_S, name_S = self.decode_data(self.files_S, index)
        image_T, _, name_T = self.decode_data(self.files_T, index)

        if self.transform is not None:
            image_S, image_T, label_S = self.transform([image_S, image_T], label_S)

        return {'Img_S': image_S, 'Img_T': image_T, \
                'Label_S': label_S, 'name_S': name_S, 'name_T': name_T}


