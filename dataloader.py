import os

import numpy as np
import scipy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageNetDataset(Dataset):

    def __init__(self, path, split, transform = None):
        super().__init__()

        img_dir_name = [v for v in os.listdir(f'{path}/{split}')]
        meta_data = scipy.io.loadmat(f'{path}/ILSVRC2012_devkit_t12/data/meta.mat')
        # save the mapping from dir_name to idx, brief_text, detailed_text
        self.dir_name_to_idx_text = {
            meta_data['synsets'][i][0][1].item(): {
                'idx': i,
                'brief_text': meta_data['synsets'][i][0][2].item(),
                'detailed_text': meta_data['synsets'][i][0][3].item()
            }
            for i in range(len(meta_data['synsets'])) if meta_data['synsets'][i][0][1] in img_dir_name
        }
        self.img = []   # save img path
        self.label = [] # save img label, brief_text, detailed_text
        for dir_name in tqdm(img_dir_name, desc = 'load ImageNet Dataset'):
            for img_name in os.listdir(f'{path}/{split}/{dir_name}'):
                self.img.append(f'{path}/{split}/{dir_name}/{img_name}')
                self.label.append(self.dir_name_to_idx_text[dir_name]['idx'])
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale = (0.08, 1.0), ratio = (3. / 4., 4. / 3.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p = 0.8),
                transforms.RandomGrayscale(p = 0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size = 3)
                ], p = 0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        with Image.open(self.img[i]) as img:
            img = img.convert('RGB')
            img = self.transform(img)
        return img, self.label[i]


if __name__ == '__main__':
    path = '/root/autodl-tmp/imagenet'
    train_dataset = ImageNetDataset(path = path, split = 'train')
    val_dataset = ImageNetDataset(path = path, split = 'val')
    print(train_dataset.dir_name_to_idx_text == val_dataset.dir_name_to_idx_text)