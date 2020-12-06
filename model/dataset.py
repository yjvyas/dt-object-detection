import os
import numpy as np
import torch
from torchvision import transforms

class DuckietownObjectDataset(object):
    def __init__(self, root):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(root)))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.imgs[idx])
        data = np.load(data_path)
        
        # convert everything into a torch.Tensor
        image = data['arr_0']/255
        image = torch.as_tensor(image, dtype=torch.float32)
        image = image.T
        image = self.normalize(image)

        boxes = torch.as_tensor(data['arr_1'], dtype=torch.int64)
        labels = torch.as_tensor(data['arr_2'], dtype=torch.int64)

        return image, {'boxes':boxes, 'labels':labels}

    def __len__(self):
        return len(self.imgs)
