import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np

from models import *
from imagenet_dataset import ImageNetDataset
from helper import AverageMeter, save_checkpoint, accuracy, adjust_learning_rate
from torch.utils.data import DataLoader
from datetime import datetime
from focalloss import FocalLoss

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

best_prec1 = 0.0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    else:
        raise NotImplementedError

    # use cuda
    model.cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = FocalLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True

    # Data loading
    train_dataset = ImageNetDataset(args.data + '/img', args.data + '/train_all.txt')
    #val_dataset = ImageNetDataset(args.data + '/img', args.data + '/test.txt')
    test_dataset = ImageNetDataset(args.data + '/img', args.data + '/test.txt', train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    eval_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    if args.evaluate:
         evaluate(eval_loader, model, criterion, args.print_freq)
         return

    for epoch in range(args.start_epoch, args.epochs):
        #adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        #prec1, prec5 = validate(val_loader, model, criterion, args.print_freq)

        # remember the best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, False, args.arch + '.pth')


def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data_dict in enumerate(train_loader):
        targets1 = data_dict['label']
        targets = torch.LongTensor([item.numpy() for item in targets1]).cuda()
        input = data_dict['image']
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda()

        # compute output
        output = model(input)
        # l0 = criterion(output[0], targets[0])
        # l1 = criterion(output[1], targets[1])
        # l2 = criterion(output[2], targets[2])
        # l3 = criterion(output[3], targets[3])
        # l4 = criterion(output[4], targets[4])
        # l5 = criterion(output[5], targets[5])
        # print(targets.device)
        loss = criterion(output, targets)


        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, data_dict in enumerate(val_loader):
        target = data_dict['label']
        input = data_dict['image']
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg

def evaluate(eval_loader, model, criterion, print_freq):

    # switch to evaluate mode
    model.eval()
    fn = datetime.now().strftime('%m-%d-%H:%M:%S')
    fn = 'prediction'
    f = open('./{}.txt'.format(fn), 'w')
    for i, data_dict in enumerate(eval_loader):
        input = data_dict['image']
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            _, pred = torch.topk(output, 1, 1, largest=True, sorted=True)
            pred = np.array(pred.squeeze().cpu())
            pred = np.swapaxes(pred, 0, 1)
            for res in pred:
                printstr = ''
                for res_single in res:
                    printstr += str(res_single) + ' '
                printstr.rstrip(' ')
                printstr += '\n'
                f.write(printstr)
    f.close()
    return


if __name__ == '__main__':
    main()
