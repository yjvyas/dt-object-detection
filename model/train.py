import os
import numpy as np
from engine import train_one_epoch, evaluate
import utils
import torch
from torchvision import transforms as T
from model import create_model
from torchvision.transforms.functional import to_tensor
import random
from PIL import Image


class DuckietownObjectDataset(object):
    def __init__(self, root):
        self.root = root
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(filter(lambda x: "npz" in x, os.listdir(root)))
        self.prob = 0.5
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        data_path = os.path.join(self.root, self.imgs[idx])
        
        data = np.load(data_path, allow_pickle=True)
        
        image = data['arr_0']
        
        boxes = torch.as_tensor(data['arr_1'], dtype=torch.float32)
        labels = torch.as_tensor(data['arr_2'], dtype=torch.int64)
        
        if random.random() < self.prob:
            height, width = image.shape[0:2]
            image = np.flip(image, 1)
            bbox = boxes
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            boxes = bbox
        
        image = to_tensor(np.copy(image))
        
        target = {'boxes':boxes, 'labels':labels}

        return image, target

    def __len__(self):
        return len(self.imgs)

def main(train_path, model_path='./model/weights/', num_epochs=5, batch_size=2, lr=0.005, weight_decay=0.0005):
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!
    print('batch size = {}'.format(batch_size))
    print('lr = {}'.format(lr))
    print('weight_decay = {}'.format(weight_decay))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = create_model()
    model.to(device)

    train_dataset = DuckietownObjectDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, \
        batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=weight_decay)
    
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, test_loader, device=device)

    model_filepath = os.path.join(model_path, 'model.pt')
    torch.save(model.state_dict(), model_filepath)
    print('Model saved to {}'.format(model_filepath))

    print("Training completed.")

if __name__ == "__main__":
    train_path = './data_collection/dataset/'
    main(train_path)