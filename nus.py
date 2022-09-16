import os
import csv
import os.path
import tarfile
from urllib.parse import urlparse
from torchvision import transforms
from src.helper_functions.helper_functions import CutoutPIL
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import re

def load_concepts(filename):
    fi=open(filename,'r')
    list1 = fi.readlines()
    dist={}
    for i in range(0, len(list1)):
        list1[i] = list1[i].strip('\n')
        dist[list1[i]]=i
    return dist
def remove(s,a):
    s=re.sub(a,'',s)
    return s
def load_image_csv(filename,nus):
    import csv
    n=0
    
    nus.images=[]
    with open(filename, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
          n+=1
          if n==1:
              continue
          filename=row[0]
          
          classes=remove(remove(row[1].strip('[').strip(']'),'\''),',').split(' ')
          
          if row[2]==nus.set:
              ar=np.zeros(len(nus.classes))
              for i in range(len(classes)):
                  id=nus.classes[classes[i]]
                  ar[id]=1
              x=[0,0]
              x[0]=filename
              x[1]=ar
              nus.images.append(x)
          
class NUS_WIDE(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        file_csv = os.path.join(root, 'nus_wid_data.csv')
        concept_file=os.path.join(root,'Concepts81.txt')
        
        self.classes = load_concepts(concept_file)
        
        load_image_csv(file_csv,self)
       
        print('[dataset] NUS classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        path=os.path.join(self.root,path)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
def get_nuswide_train(image_size,_mean,_std):
    tr=transforms.Compose([
                transforms.Resize((image_size, image_size)),
               CutoutPIL(cutout_factor=0.5),
                transforms.ToTensor(),
               transforms.Normalize(mean=_mean,
                                             std=_std),
            ])
    
    return NUS_WIDE("./NUS_WIDE","train",tr)
def get_nuswide_test(image_size,_mean,_std):
    tr=transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_mean,
                                             std=_std),
            ])
    return NUS_WIDE("./NUS_WIDE","val",tr)
