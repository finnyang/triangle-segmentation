import sys
import os
from optparse import OptionParser
import numpy as np
import numpy.random as nr

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_img = '/home/yang/triangle-segmentation/data/train/'
    dir_mask = '/home/yang/triangle-segmentation/data/masks/'
    dir_checkpoint = 'checkpoints/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0

        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs = F.sigmoid(masks_pred)
            masks_probs_flat = masks_probs.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            #print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            print loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

def train_serial(args):

    train_path='../Dataset/trainset'
    test_path='../Dataset/testset'
    ### dataset for train models


    rates=np.linspace(0,1,11)
    train_normal_dir=os.path.join(train_path,'normal')
    train_normal_mask_dir=os.path.join(train_path,'normal_mask')
    train_abnormal_dir=os.path.join(train_path,'abnormal')
    train_abnormal_mask_dir=os.path.join(train_path,'abnormal_mask')

    train_normal_id=os.listdir(train_normal_dir)
    train_normal=[] ###train set
    for i in range(len(train_normal_id)):
        train_normal.append((os.path.join(train_normal_dir,train_normal_id[i]),os.path.join(train_normal_mask_dir,train_normal_id[i])))

    train_abnormal_id=os.listdir(train_abnormal_dir)
    train_abnormal=[] ###train set
    for i in range(len(train_abnormal_id)):
        train_abnormal.append((os.path.join(train_abnormal_dir,train_abnormal_id[i]),os.path.join(train_abnormal_mask_dir,train_abnormal_id[i])))
    nr.shuffle(train_abnormal)

    ### dataset for test
    test_abnormal_dir=os.path.join(test_path,'abnormal')
    test_abnormal_mask_dir=os.path.join(test_path,'abnormal_test')
    test_abnormal_id = os.listdir(test_abnormal_dir)
    test_abnormal=[]
    for i in range(len(test_abnormal_id)):
        test_abnormal.append((os.path.join(test_abnormal_dir,test_abnormal_id[i]),os.path.join(test_abnormal_dir,test_abnormal_id[i])))

    '''
    for item in train_normal:
        if os.path.exists(item[0]) and os.path.exists(item[1]):
            continue
        else:
            print item
    for item in train_abnormal:
        if os.path.exists(item[0]) and os.path.exists(item[1]):
            continue
        else:
            print item
    for item in test_abnormal:
        if os.path.exists(item[0]) and os.path.exists(item[1]):
            continue
        else:
            print item
    '''

    for rate in rates:
        dir_checkpoint='checkpoints'+str(int(rate*10))+'/'
        if not os.path.exists(dir_checkpoint):
            os.mkdir(dir_checkpoint)
        trainset=train_normal+train_abnormal[0:int(rate*len(train_abnormal))]
        nr.shuffle(trainset)
        testset=test_abnormal


        pass





if __name__ == '__main__':
    args = get_args()

    train_serial(args)



'''



    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
'''