# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torchvision.datasets import ImageFolder

from dataset import MVTecDataset, get_data_transforms
from de_resnet import de_resnet18, de_resnet34, de_resnet50, de_wide_resnet50_2
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from test import evaluation, visualization, test

DEFAULT_CLASSES = [
    'carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood'
]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loss_fucntion(a, b):
    #mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        #print(a[item].shape)
        #print(b[item].shape)
        #loss += 0.1*mse_loss(a[item], b[item])
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss

def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        #loss += mse_loss(a[item], b[item])
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


def resolve_classes(data_root, classes_arg):
    if classes_arg:
        return [c.strip() for c in classes_arg.split(',') if c.strip()]

    if os.path.isdir(data_root):
        detected = [
            d for d in sorted(os.listdir(data_root))
            if os.path.isdir(os.path.join(data_root, d))
        ]
        if detected:
            return detected

    return DEFAULT_CLASSES

def train(_class_, data_root, epochs=200, learning_rate=0.005, batch_size=16, image_size=256):
    print(_class_)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = os.path.join(data_root, _class_, 'train')
    test_path = os.path.join(data_root, _class_)
    ckp_path = './checkpoints/' + 'wres50_'+_class_+'.pth'
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{_class_}_train_metrics.csv')
    best_path = os.path.join(log_dir, f'{_class_}_train_best.txt')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'pixel_auroc', 'sample_auroc', 'pixel_aupro', 'avg_inference_time_s'])
    best_metrics = {'pixel_auroc': 0, 'sample_auroc': 0, 'pixel_aupro': 0, 'epoch': 0}
    train_data = ImageFolder(root=train_path, transform=data_transform)
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)

    optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn.parameters()), lr=learning_rate, betas=(0.5,0.999))


    for epoch in range(epochs):
        bn.train()
        decoder.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))#bn(inputs))
            loss = loss_fucntion(inputs, outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))
        if (epoch + 1) % 10 == 0:
            auroc_px, auroc_sp, aupro_px, avg_time = evaluation(encoder, bn, decoder, test_dataloader, device)
            print('Pixel Auroc:{:.3f}, Sample Auroc{:.3f}, Pixel Aupro{:.3}, Inference Time/Image:{:.3f}s'.format(
                auroc_px, auroc_sp, aupro_px, avg_time))
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, auroc_px, auroc_sp, aupro_px, avg_time])
            if auroc_px > best_metrics['pixel_auroc'] or auroc_sp > best_metrics['sample_auroc'] or aupro_px > best_metrics['pixel_aupro']:
                best_metrics = {'pixel_auroc': auroc_px, 'sample_auroc': auroc_sp, 'pixel_aupro': aupro_px, 'epoch': epoch + 1}
                with open(best_path, 'w') as f:
                    f.write('Best metrics at epoch {}\n'.format(epoch + 1))
                    f.write('Pixel AUROC: {:.3f}\nSample AUROC: {:.3f}\nPixel AUPRO: {:.3f}\n'.format(auroc_px, auroc_sp, aupro_px))
            torch.save({'bn': bn.state_dict(),
                        'decoder': decoder.state_dict()}, ckp_path)
    return auroc_px, auroc_sp, aupro_px, avg_time




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=os.getenv('DATA_ROOT', './mvtec'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--classes', default=None,
                        help='Comma-separated class names. If omitted, classes are autodetected from data_root.')
    args = parser.parse_args()

    setup_seed(111)
    item_list = resolve_classes(args.data_root, args.classes)
    print('Training classes:', ', '.join(item_list))
    for i in item_list:
        train(i, args.data_root, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch_size,
              image_size=args.image_size)

