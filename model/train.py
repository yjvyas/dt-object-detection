import os
from engine import train_one_epoch, evaluate
from dataset import DuckietownObjectDataset
import utils
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def main(train_path, test_path, model_path='./model/weights/', num_epochs=10):
    # TODO train loop here!
    # TODO don't forget to save the model's weights inside of `./weights`!
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)
    model.to(device)

    train_dataset = DuckietownObjectDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, \
        batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)

    test_dataset = DuckietownObjectDataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_dataset, \
        batch_size=2, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, test_loader, device=device)

    model_name = 'model_{}.pth'.format(epochs)
    torch.save(model.state_dict(), os.path.join(model_path, model_name))

    print("Training completed.")

if __name__ == "__main__":
    train_path = './data_collection/dataset/train_dataset/'
    test_path = './data_collection/dataset/validation_dataset/'
    main(train_path, test_path)