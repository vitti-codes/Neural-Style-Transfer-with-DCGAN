from PIL import Image
from torch.utils.data.dataset import Dataset
import torch
import os
from skimage import io
import pandas as pd
import chardet



class wikiart_dataset(Dataset):
    def __init__(self, csv_file,  root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, encoding='ISO-8859-1', index_col=None, header=None)
        self.annotations.columns = ['path', 'n']
        self.root_dir = root_dir
        self.transform = transform

    def __getitem__(self, index):
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        #image = io.imread(img_path)
        #print(os.getcwd())
        img_path = self.annotations.iloc[index]['path']
        img_path = img_path.replace('./Impressionism/', 'Impressionism/')
        image = Image.open(img_path)
        #y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        y_label = torch.tensor(int(self.annotations.iloc[index]['n']))

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.annotations)
        #return count (of how many examples/images you have
