from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import numpy as np
import torch
from torchvision import transforms
from .utils import download_url, check_integrity

transform_usual = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    training_file = 'training.pt'
    training_extra_file = 'training_extra.pt'
    train_split_file = 'train-split.pt'
    val_split_file = 'val-split.pt'
    test_file = 'test.pt'
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, run_folder='run99', train=False, val=False, extra=True, transform=None, target_transform=None, download=False, train_list=[], imbalance_ratio=1, noisy_ratio=0.0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # train set
        self.val = val  # val set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if not self._split_exists():
            raise RuntimeError('Train/val split not found.' +
                               ' You can use download=True to download it')

        if self.val:
            data_file = '{}/{}'.format(self.processed_folder, self.val_split_file)
        elif self.train:
            if (imbalance_ratio==1) and (noisy_ratio==0.0):
                data_file = '{}/{}'.format(self.processed_folder, self.train_split_file)
            else:
                data_file = '{}/{}/data/train_ir_{}_nr_{}.pt'.format(root, run_folder, imbalance_ratio, noisy_ratio)
        else:
            data_file = '{}/{}'.format(self.processed_folder, self.test_file)
        
        data, targets = torch.load(data_file)

        print('Original dataset', data_file, data.size(), targets.size())
        if (self.train==True) and (len(train_list)>0):
            data, targets = data[train_list], targets[train_list]
            print('Subsampled dataset', data_file, data.size(), targets.size())
        

        self.data, self.targets = data, targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.test_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.training_extra_file))
                )

    def _split_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.train_split_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.val_split_file))
                )

    def _check_integrity(self):
        for fentry in ['train', 'test', 'extra']:
            filename, md5 = self.split_list[fentry][1], self.split_list[fentry][2]
            fpath = os.path.join(self.raw_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        """Download the SVHN data if it doesn't exist in processed_folder already."""

        if self._check_exists() and self._split_exists():
            print('Files already processed')
            return

        if self._check_integrity():
            print('Files already downloaded and verified')
        else:
            print('Files are not manually downloaded!')
            for fentry in ['train', 'test', 'extra']:
                [url, filename, md5] = self.split_list[fentry]
                download_url(url, self.raw_folder, filename, md5)

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio


        if not os.path.exists(self.processed_folder):
            os.makedirs(self.processed_folder)

        # test dataset
        loaded_mat = sio.loadmat(os.path.join(self.raw_folder, self.split_list['test'][1]))
        data = loaded_mat['X']
        data = np.transpose(data, (3, 0, 1, 2))
        print('TEST DATA', data.shape)

        labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(labels, labels == 10, 0)
        print('TEST LABELS', labels.shape)

        processed_set = (
            torch.from_numpy(data),
            torch.tensor(labels)
        )

        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(processed_set, f)

        # train dataset
        loaded_mat = sio.loadmat(os.path.join(self.raw_folder, self.split_list['train'][1]))
        data = loaded_mat['X']
        data = np.transpose(data, (3, 0, 1, 2))
        print('TRAIN DATA', data.shape)

        labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(labels, labels == 10, 0)
        print('TRAIN LABELS', labels.shape)

        processed_set = (
            torch.from_numpy(data),
            torch.tensor(labels)
        )

        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(processed_set, f)
    
        # train extra dataset
        loaded_mat = sio.loadmat(os.path.join(self.raw_folder, self.split_list['extra'][1]))
        extra_data = loaded_mat['X']
        extra_data = np.transpose(extra_data, (3, 0, 1, 2))
        data = np.concatenate((data, extra_data), axis=0)
        print('TRAIN EXTRA DATA', data.shape)

        extra_labels = loaded_mat['y'].astype(np.int64).squeeze()
        np.place(extra_labels, extra_labels == 10, 0)
        labels = np.concatenate((labels, extra_labels), axis=0)
        print('TRAIN EXTRA LABELS', labels.shape)

        processed_set = (
            torch.from_numpy(data),
            torch.tensor(labels)
        )

        with open(os.path.join(self.processed_folder, self.training_extra_file), 'wb') as f:
            torch.save(processed_set, f)

        # splitting full train dataset into train/val splits
        V = 104388 # val size
        C = 10 # 10 classes
        VC = int(V/C+0.5)
        data, targets = processed_set

        vImages = torch.tensor([], dtype=torch.uint8)
        vLabels = torch.tensor([], dtype=torch.long)
        tImages = torch.tensor([], dtype=torch.uint8)
        tLabels = torch.tensor([], dtype=torch.long)
        for c in range(C):
            mask = targets.eq(c)
            index = mask.nonzero().squeeze().tolist()
            if (c > 7): # wanna get exactly 500K training dataset
                vIdx = random.sample(index, VC-1) # select randomly sunbset
            else:
                vIdx = random.sample(index, VC) # select randomly sunbset
            tIdx = [item for item in index if item not in vIdx]
            print(c, ':', len(index),'=', len(vIdx)+len(tIdx))
            vImages = torch.cat([vImages, torch.index_select(data,    0, torch.tensor(vIdx, dtype=torch.long))])
            vLabels = torch.cat([vLabels, torch.index_select(targets, 0, torch.tensor(vIdx, dtype=torch.long))])
            tImages = torch.cat([tImages, torch.index_select(data,    0, torch.tensor(tIdx, dtype=torch.long))])
            tLabels = torch.cat([tLabels, torch.index_select(targets, 0, torch.tensor(tIdx, dtype=torch.long))])

        with open(os.path.join(self.processed_folder, self.val_split_file), 'wb') as f:
            torch.save((vImages, vLabels), f)

        with open(os.path.join(self.processed_folder, self.train_split_file), 'wb') as f:
            torch.save((tImages, tLabels), f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
