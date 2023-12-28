import torch
import argparse
import time
import os
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.tensorboard.writer import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.cuda.amp import autocast as autocast
from torch.nn.modules.loss import CrossEntropyLoss
from loss import LovaszLoss, DiceLoss, TverskyLoss
from net import UNet, init_weights
from ResUnet import DeepResUNet
from DenseUnet import DenseUnet
from tqdm import tqdm
from dataset import ImageFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args, model, criterion, optimizer, dataloaders):
    # ce_loss, dice_loss = criterion
    model.train()
    writer = SummaryWriter(args.log_path + '/writer')
    iter_num = 0
    iterator = tqdm(range(args.max_epochs), ncols=70)
    for epoch in iterator:
        epoch_loss = 0
        for i_batch, sampled_batch in enumerate(dataloaders):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            optimizer.zero_grad()
            outputs = model(image_batch)

            # criterion
            voutputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            vlabel_batch = label_batch.view(-1).long()
            # loss_ce = ce_loss(voutputs, vlabel_batch[:].long())

            # loss_dice = dice_loss(outputs, label_batch.squeeze(1), softmax=True)
            # loss = 0.4 * loss_ce + 0.6 * loss_dice
            loss = criterion(outputs, label_batch, False)
            # with autocast():

            loss.backward()
            optimizer.step()
            # scheduler.step(train_loss)
            epoch_loss += loss.item()
            print(f"{epoch}/{args.max_epochs} epoch_loss:{epoch_loss}, loss:{loss}")

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', args.learning_rate, iter_num)
            writer.add_scalar('info/loss', loss, iter_num)
            ...

        save_interval = args.save_interval
        if (epoch % save_interval == 0) and (epoch > 0):
            save_mode_path = args.ckpt
            if not os.path.exists(save_mode_path):
                os.makedirs(save_mode_path)
            torch.save(model.state_dict(), os.path.join(save_mode_path, "epoch_" + str(epoch) + ".pth"))


def val():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r'C:\Users\Administrator\Desktop\test',
                        help='root dir for data')
    parser.add_argument('--dataset', type=str, default='test', help='dataset dir')
    parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
    parser.add_argument('--max_epochs', type=int, default=1500, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--img_channel', type=int, default=3, help='input patch size of network input')
    parser.add_argument('--img_size', type=int, default=112, help='input patch size of network input')
    parser.add_argument('--ckpt', type=str, default='./model/epoch_200.pth', help='pre trained model')
    parser.add_argument('--log_path', type=str, default='./log', help='pre trained model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--save_interval', type=int, default=50, help='save interval')

    args = parser.parse_args()
    # model = DeepResUNet(img_ch=args.img_channel, output_ch=args.num_classes)
    model = DenseUnet(img_ch=args.img_channel, output_ch=args.num_classes)
    if torch.cuda.is_available():
        model = model.to(device)

    if os.path.exists(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage.cuda(0))
        model.load_state_dict(checkpoint)
        model.cuda()

    model.eval()
    sz = (args.img_size, args.img_size)

    # 图像变换
    img_transforms = transforms.Compose([
        transforms.Resize(sz),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 加载数据
    datasets = ImageFold(args.root_path, args.img_size, mode=args.dataset, transform_img=img_transforms)
    dataloaders = DataLoader(datasets, batch_size=args.batch_size, shuffle=False, num_workers=1)

    for i_batch, sampled_batch in enumerate(dataloaders):
        image_batch, label_batch, path = sampled_batch['image'], sampled_batch['label'], sampled_batch['path']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        outputs = model(image_batch)
        outputs = torch.argmax(outputs, dim=1)
        sImage = outputs.squeeze(0).data.cpu().numpy()

        images = cv2.imread(path[0])
        sImage = cv2.resize(sImage, dsize=(images.shape[1], images.shape[0]), interpolation=cv2.INTER_NEAREST)
        sImage = sImage.astype(np.uint8)
        contours, hierarchy = cv2.findContours(sImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(images, contours, -1, (0, 0, 255))

        # n_img_y = sImage.astype(np.uint8)

        cv2.namedWindow("images", cv2.WINDOW_NORMAL)
        cv2.imshow('images', images)

        # cv2.namedWindow("image_out", cv2.WINDOW_NORMAL)
        # cv2.imshow('image_out', n_img_y)
        cv2.waitKey()
    ...


if __name__ == '__main__':
    # val()
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default=r'F:\1workdata\2labeledImg\jutian\jutianImage\mask1',
                         help='root dir for data')
    parser.add_argument('--dataset', type=str, default='train', help='dataset dir')
    parser.add_argument('--num_classes', type=int, default=3, help='output channel of network')
    parser.add_argument('--max_epochs', type=int, default=1500, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--img_channel', type=int, default=3, help='input patch size of network input')
    parser.add_argument('--img_size', type=int, default=112, help='input patch size of network input')
    parser.add_argument('--ckpt', type=str, default='./model', help='pre trained model')
    parser.add_argument('--log_path', type=str, default='./log', help='pre trained model')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--save_interval', type=int, default=50, help='save interval')
    
    rgs = parser.parse_args()
    # 设定loss
    # # criterion = CrossEntropyLoss()
    # # ce_loss = CrossEntropyLoss()
    # # dice_loss = TverskyLoss(args.num_classes)
    #
    criterion = LovaszLoss()
    # # criterion = (ce_loss, dice_loss)
    # # 加载模型，单个gpu
    # # model = DeepResUNet(img_ch=args.img_channel, output_ch=args.num_classes)
    # # init_weights(model, init_type="kaiming")
    model = DenseUnet(img_ch=args.img_channel, output_ch=args.num_classes)
    if torch.cuda.is_available():
         model = model.to(device)
    
    # # checkpoints = os.listdir(args.ckpt)
    # # if len(checkpoints) > 0:
    # #     model.load_state_dict(torch.load(os.path.join(args.ckpt, checkpoints[len(checkpoints) - 1])))
    # # 设定优化
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    # sz = (args.img_size, args.img_size)
    #
    # 图像变换
    img_transforms = transforms.Compose([
        transforms.Resize(sz),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
     ])
    
     label_transforms = transforms.Compose([
         transforms.Resize(sz, interpolation=InterpolationMode.NEAREST),
         transforms.ToTensor()
     ])
    
     # 加载数据
     datasets = ImageFold(args.root_path, args.img_size, mode=args.dataset, transform_img=img_transforms)
     dataloaders = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=4)
     train(args, model, criterion, optimizer, dataloaders)

