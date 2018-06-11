import torch
import torch.utils.data
import os
import csv
import errno
import scipy.io

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

import google_drive

from PIL import Image

class MNIST_M(torch.utils.data.Dataset):
    """MNIST_M dataset."""
    
    data_dir = "mnist_m/"
    file_id = ["1-5-DRC8ZJ6TYUDP0CA0vbwElJL3z1hHT", "14OVKf4nrYVlD-kNmqYrEhsvD5Ty_E9Ya"]
    raw_file = ["train.pt.gz", "test.pt.gz"]
    Tensor2PIL = transforms.ToPILImage()

    def __init__(self, root_dir, train=True, transform=None, download=False):
        """
        Args:
            root_dir (string): Path to mnist_m directory.
            train (bool, optional): If True, create training set, otherwise
                test set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")
            
        if self.train:
            data_file = "train.pt"
        else:
            data_file = "test.pt"
            
        self.images, self.labels = torch.load(
            os.path.join(self.root_dir, self.data_dir, data_file)
        )

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = self.Tensor2PIL(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root_dir, self.data_dir, "train.pt")) and \
            os.path.exists(os.path.join(self.root_dir, self.data_dir, "test.pt"))
    
    def download(self):
        import gzip
        
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root_dir, self.data_dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for idx in range(len(self.file_id)):
            print("Downloading " + self.raw_file[idx])
            file_path = os.path.join(self.root_dir, self.data_dir, self.raw_file[idx])
            google_drive.download_file_from_google_drive(self.file_id[idx], file_path)
            
            print("Extracting " + self.raw_file[idx])
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
            
        print("Done.")
            
class SYNNUM(torch.utils.data.Dataset):
    """Synthetic numbers dataset."""
    
    data_dir = "synnum/"
    file_id = "0B9Z4d7lAwbnTSVR1dEFSRUFxOUU"
    raw_file = "SynthDigits.zip"

    def __init__(self, root_dir, train=True, small=False, transform=None, download=False):
        """
        Args:
            root_dir (string): Path to mnist_m directory.
            train (bool, optional): If True, create training set, otherwise test set.
            small (bool, optional): If True, read the small dataset, otherwise read the
                original dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.root_dir = root_dir
        
        self.data_file = "synth"
        if train:
            self.data_file += "_train_32x32"
        else:
            self.data_file += "_test_32x32"
        if small:
            self.data_file += "_small"
        self.data_file += ".mat"
        
        self.transform = transform
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")
            
        mat = scipy.io.loadmat(os.path.join(self.root_dir, self.data_dir, self.data_file))
        self.images = mat["X"]
        self.labels = torch.LongTensor(mat["y"]).view(-1)


    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        print(idx)
        image, label = self.images[:, :, :, idx], self.labels[idx]
        
        image = Image.fromarray(image, mode="RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root_dir, self.data_dir, self.data_file))
    
    def download(self):
        import zipfile
        
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root_dir, self.data_dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        print("Downloading " + self.raw_file)
        file_path = os.path.join(self.root_dir, self.data_dir, self.raw_file)
        google_drive.download_file_from_google_drive(self.file_id, file_path)
        
        print("Extracting " + self.raw_file)
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(os.path.join(self.root_dir, self.data_dir))
        zip_ref.close()
        os.unlink(file_path)
        
        print("Done.")

class ST_Dataset(torch.utils.data.Dataset):
    """Source and target dataset combination."""
    
    def __init__(self, source, target, batch_size):
        """
        Args:
            source (torch.utils.data.Dataset): Source dataset.
            target (torch.utils.data.Dataset): Target dataset.
            batch_size (int): Batch size.
        """
        small_len = min(len(source), len(target))
        self.length = small_len * 2
        channel, height, width = source[0][0].size()
        self.images = torch.Tensor(small_len * 2, channel, height, width)
        self.labels = torch.LongTensor(small_len * 2)
        self.domains = torch.LongTensor(small_len * 2)
        start = 0
        half_batch_size = batch_size // 2
        cnt = 0
        while start < small_len:
            end = min(start + half_batch_size, small_len)
            for i in range(start, end):
                # source
                self.images[start * 2 + (i - start)] = source[i][0]
                self.labels[start * 2 + (i - start)] = source[i][1]
                self.domains[start * 2 + (i - start)] = 0
                # target
                self.images[start * 2 + (end - start) + (i - start)] = target[i][0]
                self.labels[start * 2 + (end - start) + (i - start)] = -1
                self.domains[start * 2 + (end - start) + (i - start)] = 1
            start += half_batch_size
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.domains[idx]

    
# download preprocessing data from synthetic sign dataset
# https://graphics.cs.msu.ru/en/node/1337
class SYNSIGN(torch.utils.data.Dataset):
    """Synthetic numbers dataset."""
    
    data_dir = "synsign/"
    file_id = "1wkExFXkv22byHJrq9J9trenJrhSX6CRP"
    raw_file = "synsign.zip"
    
    def __init__(self, root_dir, train=True, small=False, transform=None, download=False):
        """
        Args:
            root_dir (string): Path to mnist_m directory.
            train (bool, optional): If True, create training set, otherwise test set.
            small (bool, optional): If True, read the small dataset, otherwise read the
                original dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.root_dir = root_dir
        
        self.data_file = "synsign.zip"
        
        self.transform = transform
        
        if download:
            self.download()
            
        #if not self._check_exists():
        #    raise RuntimeError("Dataset not found." +
        #                       " You can use download=True to download it")
            
        
        import numpy as np
        
        self.images = np.load(os.path.join(self.root_dir, self.data_dir, 'synsign/img_data.npy'))
        
        self.images = np.transpose(self.images, (2, 0, 1, 3)).astype(float)
        self.labels = torch.LongTensor(np.load(os.path.join(self.root_dir, self.data_dir, 'synsign/label.npy')))
        self.labels = self.labels.view(-1)


    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        #print(idx)
        image, label = self.images[:, :, :, idx], self.labels[idx]
        
        #image = Image.fromarray(image, mode="RGB")

        #if self.transform is not None:
        #    image = self.transform(image)
        
        return image, label
    
    def _check_exists(self):

        return os.path.exists(os.path.join(self.root_dir, self.data_dir, self.data_file))
    
    def download(self):
        import zipfile
        
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root_dir, self.data_dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        print("Downloading " + self.raw_file)
        file_path = os.path.join(self.root_dir, self.data_dir, self.raw_file)
        google_drive.download_file_from_google_drive(self.file_id, file_path)

        print("Extracting " + self.raw_file)
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(os.path.join(self.root_dir, self.data_dir))
        zip_ref.close()
        os.unlink(file_path)
        
        print("Done.")    

# download preprocessing data from german traffic signs dataset
# http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
class GTSRB(torch.utils.data.Dataset):
    """Synthetic numbers dataset."""
    
    data_dir = "GTSRB/"
    file_id = "1WkenB2nR7U9BzHoIxbuSWYHHh5lJZT4L"
    raw_file = "GTSRBv2.zip"
    
    def __init__(self, root_dir, train=True, small=False, transform=None, download=False):
        """
        Args:
            root_dir (string): Path to mnist_m directory.
            train (bool, optional): If True, create training set, otherwise test set.
            small (bool, optional): If True, read the small dataset, otherwise read the
                original dataset.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        """
        self.root_dir = root_dir
        
        self.data_file = "GTSRBv2.zip"
        
        self.transform = transform
        
        if download:
            self.download()
            
        #if not self._check_exists():
        #    raise RuntimeError("Dataset not found." +
        #                       " You can use download=True to download it")
            
        
        import numpy as np
        self.train = train
        if self.train:
            self.images = np.load(os.path.join(self.root_dir, self.data_dir, 'GTSRBv2/train_imgv2.npy'))
            self.images = np.transpose(self.images, (2, 0, 1, 3)).astype(float)
            self.labels = torch.LongTensor(np.load(os.path.join(self.root_dir, self.data_dir, 'GTSRBv2/train_labelv2.npy')))
            self.labels = self.labels.view(-1)
        else:
            self.images = np.load(os.path.join(self.root_dir, self.data_dir, 'GTSRBv2/test_img.npy'))
            self.images = np.transpose(self.images, (2, 0, 1, 3)).astype(float)
            self.labels = torch.LongTensor(np.load(os.path.join(self.root_dir, self.data_dir, 'GTSRBv2/test_label.npy')))
            self.labels = self.labels.view(-1)



    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        #print(idx)
        image, label = self.images[:, :, :, idx], self.labels[idx]
        
        #image = Image.fromarray(image, mode="RGB")

        #if self.transform is not None:
        #    image = self.transform(image)
        
        return image, label
    
    def _check_exists(self):

        return os.path.exists(os.path.join(self.root_dir, self.data_dir, self.data_file))
    
    def download(self):
        import zipfile
        
        if self._check_exists():
            return

        try:
            os.makedirs(os.path.join(self.root_dir, self.data_dir))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        
        print("Downloading " + self.raw_file)
        file_path = os.path.join(self.root_dir, self.data_dir, self.raw_file)
        google_drive.download_file_from_google_drive(self.file_id, file_path)

        print("Extracting " + self.raw_file)
        zip_ref = zipfile.ZipFile(file_path, "r")
        zip_ref.extractall(os.path.join(self.root_dir, self.data_dir))
        zip_ref.close()
        os.unlink(file_path)
        
        print("Done.")    
        
        
def txt_to_csv(i, o):
    in_txt = csv.reader(open(i, "r"), delimiter = ' ')
    out_csv = csv.writer(open(o, "w"))
    out_csv.writerows(in_txt)
