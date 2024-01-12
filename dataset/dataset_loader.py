from PIL import Image
from tqdm import tqdm

import os
import multiprocessing as mp

try:
    from dataset import utils
except:
     import utils


# Dataset
class dataset_loader:
    def __init__(self, root, task, is_train=False, transforms=None, in_memory=False):
        self.root = root
        self.task = task
        self.is_train = is_train
        self.transforms = transforms
        self.in_memory = in_memory

        self.img_path, self.label = getattr(utils, task)(root, is_train)
        self.count, self.n_class = statistic(self.img_path, self.label)

        print('There are {} {} images, {} categories.'.format(
               len(self.label), 
               'train' if is_train else 'validation', 
               self.n_class, 
               ))

        if in_memory:
            print('Pre-load the images into memory ...')
            pool = mp.Pool()
            N = len(self.img_path)
            pbar = tqdm(total=N)
            def update(*a):
                pbar.update()
            results = []
            for row in self.img_path:
                results.append(pool.apply_async(load_image, args=(row,), callback=update))
            pool.close()
            pool.join()
            pbar.close()
            self.image = [row.get() for row in results]
        
    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = load_image(f)

        label = self.label[index]
        sample = self.transforms(sample)

        return sample, label, path

def load_image(path):
    return Image.open(path).convert('RGB')

def statistic(img_path, label):
    id2img = {}
    for img, l in zip(img_path, label):
        id2img.setdefault(l, [])
        id2img[l].append(img)

    keys = list(id2img.keys())
    keys.sort()
    count = [len(id2img[key]) for key in keys]
    n_class = len(count)
    return count, n_class

if __name__ == '__main__':
    import torch
    from augmentation import augmentation

    def run(root, task, is_train=True):
        transforms = augmentation(resize=256, size=224, is_train=is_train)
        dataset = dataset_loader(root=root, task=task, is_train=is_train, transforms=transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, 
                                                 shuffle=False, num_workers=16, drop_last=False)

        pbar = tqdm(dataloader)
        for image, label, path in pbar:
            pbar.set_description('image: {}, label: {}'.format(list(image.size()), 
                                                               list(label.size()),
                                                               ))

    root = '/work/v20180902/dataset'
    is_train = True
    tasks = ['inaturalist']
    for t in tasks:
        run(root, t, is_train)
