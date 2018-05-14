import torch
import torch.utils.data
import pandas as pd
import os
from PIL import Image
import csv
import google_drive
import errno
import torchvision.transforms as transforms

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
        self.images = torch.Tensor(small_len * 2, 3, 28, 28)
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
    
def txt_to_csv(i, o):
    in_txt = csv.reader(open(i, "r"), delimiter = ' ')
    out_csv = csv.writer(open(o, "w"))
    out_csv.writerows(in_txt)
