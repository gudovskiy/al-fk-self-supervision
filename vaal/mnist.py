from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import gzip
import codecs
import numpy as np
import torch
from torchvision import transforms
from utils import download_url, makedir_exist_ok


def mnist_transformer():
    return transforms.Compose([
           transforms.Pad(2),
           transforms.ToTensor()#,
           #transforms.Normalize((0.1307,), (0.3081,))
    ])


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    train_file = 'training.pt'
    train_split_file = 'train-split.pt'
    val_split_file = 'val-split.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, run_folder='run99', train=False, val=False, transform=mnist_transformer(), target_transform=None, download=False, train_list=[], imbalance_ratio=1, noisy_ratio=0.0):
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

        if self.val and not(self.train):
            data_file = '{}/{}'.format(self.processed_folder, self.val_split_file)
            data, targets = torch.load(data_file)
        elif self.train:
            if (imbalance_ratio==1) and (noisy_ratio==0.0):
                data_file = '{}/{}'.format(self.processed_folder, self.train_split_file)
            else:
                data_file = '{}/{}/data/train_ir_{}_nr_{}.pt'.format(root, run_folder, imbalance_ratio, noisy_ratio)
            #
            data, targets = torch.load(data_file)
            if len(train_list) > 0: # select by index
                data, targets = data[train_list], targets[train_list]
            #
            if self.val: # add val data
                data_file = '{}/{}'.format(self.processed_folder, self.val_split_file)
                val_data, val_targets = torch.load(data_file)
                data, targets = torch.cat([val_data, data]), torch.cat([val_targets, targets])
        else:
            data_file = '{}/{}'.format(self.processed_folder, self.test_file)
            data, targets = torch.load(data_file)
        #
        print('Dataset {} of size {}/{}'.format(data_file, data.size(), targets.size()))
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
        img = Image.fromarray(img.numpy(), mode='L')

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

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return  os.path.exists(os.path.join(self.processed_folder, self.train_file)) and \
                os.path.exists(os.path.join(self.processed_folder, self.test_file))

    def _split_exists(self):
        return  os.path.exists(os.path.join(self.processed_folder, self.train_split_file)) and \
                os.path.exists(os.path.join(self.processed_folder, self.val_split_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists() and self._split_exists():
            print('Files already processed')
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        train_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.train_file), 'wb') as f:
            torch.save(train_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        # splitting full train dataset into train/val splits
        V = 10000 # val size
        C = len(self.classes) # 10 classes
        VC = int(V/C+0.5)
        data, targets = train_set

        vImages = torch.tensor([], dtype=torch.uint8)
        vLabels = torch.tensor([], dtype=torch.long)
        tImages = torch.tensor([], dtype=torch.uint8)
        tLabels = torch.tensor([], dtype=torch.long)
        for c in range(C):
            mask = targets.eq(c)
            index = mask.nonzero().squeeze().tolist()
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
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)

