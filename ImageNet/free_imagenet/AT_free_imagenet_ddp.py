import argparse
import os
import sys
sys.path.append("..")
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models_imagenet import *
from utils.data_loader import data_loader_imagenet_dataset
from utils.core import PGD
from utils.helper import AverageMeter, accuracy, adjust_learning_rate_free, save_checkpoint, set_seed

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from managpu import GpuManager
my_gpu = GpuManager()
using_gpu = my_gpu.set_by_memory(4)
print("Using GPU: ", using_gpu)

model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet18_hfs', 'resnet34_hfs', 'resnet50_hfs', 'resnet101_hfs',
    'resnet152_hfs','resnet18_hfs_canny', 'resnet34_hfs_canny', 'resnet50_hfs_canny', 'resnet101_hfs_canny',
    'resnet34_hfs_canny'
]


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default='/hdd/public_data/ImageNet/seqres',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet152', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', default=True, dest='pin_memory', action='store_true',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', default=False, dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=100, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint, (default: None)')
parser.add_argument('-e', '--evaluate',  dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--epsilon', type=float, default = 4.0 / 255,
                    help='perturbation')
parser.add_argument('--num-steps-1', type=int, default = 50,
                    help='perturb number of steps')
parser.add_argument('--step-size-1',type=float, default = 1.0 / 255,
                    help='perturb step size')
# parser.add_argument('--num-steps-2', type=int, default = 20,
#                     helpc='perturb number of steps')
# parser.add_argument('--step-size-2', type=float, default = 1.0 / 255,
#                     help='perturb step size')
# parser.add_argument('--num-steps-3', type=int, default = 40,
#                     help='perturb number of steps')
# parser.add_argument('--step-size-3', type=float, default = 1.0 / 255,
#                     help='perturb step size')
parser.add_argument('--random', default=True,
                    help='random initialization for PGD')

parser.add_argument('--cize', default= 224 , type=int, help='dimention of image')
parser.add_argument('--alpha', type=float, default = 0,
                        help='gradient mask')
parser.add_argument('--sigma', type=float, default = 0,
                        help='guassian of sigma')

parser.add_argument('--clip-eps', default=4.0, type=float, help='clip-eps')
parser.add_argument('--fgsm-step', default=4.0, type=float, help='fgsm-step')
parser.add_argument('--max-color-value', default=255.0, type=float, help='max color value')
parser.add_argument('--crop-size', default=224, type=int, help='Crop size of imagenet')
parser.add_argument('--n-repeats', default=4, type=int, help='Number of repeats for free adversarial training')

parser.add_argument('--w', default= 0 , type=float, help='wight of canny')
parser.add_argument('--r', default= 0 , type=int, help='radius of our supression module')
parser.add_argument('--gf', default= False, action='store_true', help='gaussian filter after canny')
parser.add_argument('--low', default= 0 , type=float, help='low_threshold of canny')
parser.add_argument('--high', default= 0 , type=float, help='high_threshold of canny')

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--nGPU', default=4, type=int, help='number of GPUs')

best_prec1 = 0.0
args = parser.parse_args()

def main():
    global args, best_prec1
    # args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # set_seed(args.seed)
    # device = torch.device("cuda" if use_cuda else "cpu")

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    device = torch.device("cuda", args.local_rank)

    rank = torch.distributed.get_rank()
    set_seed(args.seed+rank)


    args.epochs = int(math.ceil(args.epochs / args.n_repeats))
    args.fgsm_step /= args.max_color_value
    args.clip_eps /= args.max_color_value
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'resnet50':
        model = resnet50()
    elif args.arch ==  'resnet101':
        model = resnet101()
    elif args.arch ==  'resnet152':
        model = resnet152()
    elif args.arch ==  'resnet18':
        model = resnet18()
    else:
        raise NotImplementedError

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)

    # define loss and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(args.local_rank)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    if args.local_rank == 0:
        print('r:{},w:{},gf:{},low:{},high:{},alpha:{},sigma:{}'.format(args.r, args.w, args.gf, args.low, args.high,
                                                                        args.alpha, args.sigma))
        print(
            'arch:{},bs:{},lr:{},wd:{},momentum:{},epochs:{}'.format(args.arch, args.batch_size, args.lr, args.weight_decay, args.momentum,
                                                             args.epochs))
        print('clip-eps:{},fgsm-step:{},n-repeats:{},nGPU:{}'.format(int(args.clip_eps*255), int(args.fgsm_step*255),
                                                                     args.n_repeats, args.nGPU))

    # dir setting
    cur_dir = os.getcwd()
    dir = cur_dir + '/checkpoint_free_imagenet/free_AT_ddp/' + str(args.arch) + \
          'Baseline' +  '_clip-eps' +  str(int(args.clip_eps*255)) + '/'
    print(dir)
    model_dir = dir + 'model_pth/'
    best_model_dir = dir + 'best_model_pth/'
    log_dir = dir + 'log/'
    if not os.path.exists(log_dir):
        if args.local_rank == 0:
            os.makedirs(log_dir)
    if not os.path.exists(model_dir):
        if args.local_rank == 0:
            os.makedirs(model_dir)
    if not os.path.exists(best_model_dir):
        if args.local_rank == 0:
            os.makedirs(best_model_dir)


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.local_rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading
    # train_loader, val_loader = data_loader_imagenet(args.data, args.batch_size, args.workers, args.pin_memory)
    train_dataset, val_dataset = data_loader_imagenet_dataset(args.data)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.batch_size / args.nGPU), shuffle=False,
                                               num_workers=args.workers, pin_memory=args.pin_memory,
                                               sampler=train_sampler)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=int(args.batch_size / args.nGPU), shuffle=False,
                                             num_workers=args.workers, pin_memory=args.pin_memory,
                                             sampler=val_sampler)
    if args.evaluate:
        validate(val_loader, model, criterion, args.print_freq, device, log_dir)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate_free(optimizer, epoch, args.lr,args.n_repeats)

        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq, device, log_dir)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion, args.print_freq, device, log_dir)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if args.local_rank == 0:
            save_checkpoint(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict()
                },
                is_best,
                model_dir + 'at_clip-eps' +  str(int(args.clip_eps*255)) + '_fgsm-step' + str(int(args.fgsm_step*255)) +
                '_n-repeats' + str(args.n_repeats)+ '_r' + str(args.r) +
                '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
                '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
                '-w' + str(args.w) + '-gf' + str(args.gf) +
                '-l' + str(args.low)+ '-h' + str(args.high)+'-ty1_' + str(epoch) + '.pth',
                best_model_dir + 'at_clip-eps' +  str(int(args.clip_eps*255)) + '_fgsm-step' + str(int(args.fgsm_step*255)) +
                '_n-repeats' + str(args.n_repeats)+ '_r' + str(args.r) +
                '_canny_sigma' + str(args.sigma) + '_alpha' + str(args.alpha) +
                '-bs' + str(args.batch_size) + '-lr_' + str(args.lr) +
                '-w' + str(args.w) + '-gf' + str(args.gf) +
                '-l' + str(args.low)+ '-h' + str(args.high)+'-ty1_' + '.pth'
        )


global global_noise_data
# global_noise_data = torch.zeros([args.batch_size, 3, args.crop_size, args.crop_size]).cuda()
global_noise_data = torch.zeros([args.batch_size, 3, args.crop_size, args.crop_size])

def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, log_dir):
    global global_noise_data
    global_noise_data = global_noise_data.to(device)
    # mean = torch.Tensor(np.array([0.485, 0.456, 0.406])[:, np.newaxis, np.newaxis])
    # mean = mean.expand(3, args.crop_size, args.crop_size).cuda()
    # std = torch.Tensor(np.array([0.229, 0.224, 0.225])[:, np.newaxis, np.newaxis])
    # std = std.expand(3, args.crop_size, args.crop_size).cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        target = target.to(device)
        input = input.to(device)
        data_time.update(time.time() - end)
        for j in range(args.n_repeats):
            noise_batch = Variable(global_noise_data[0:input.size(0)], requires_grad=True).to(device)
            # noise_batch = Variable(global_noise_data[0:input.size(0)].to(device), requires_grad=True)
            in1 = input + noise_batch
            in1.clamp_(0, 1.0)
            # in1.sub_(mean).div_(std)
            output = model(in1)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            # Update the noise for the next iteration
            # pert = fgsm(noise_batch.grad, configs.ADV.fgsm_step)
            pert = args.fgsm_step * torch.sign(noise_batch.grad)
            global_noise_data[0:input.size(0)] += pert.data
            global_noise_data.clamp_(-args.clip_eps, args.clip_eps)

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        if i % print_freq == 0 and args.local_rank == 0:
            print_content = 'Epoch: [{0}][{1}/{2}]\t'\
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                                epoch, i, len(train_loader), batch_time=batch_time,
                                data_time=data_time, loss=losses, top1=top1, top5=top5)

            print(print_content)
            # with open(log_dir + 'log.txt', 'a') as f:
            #     print(print_content, file=f)

def validate(val_loader, model, criterion, print_freq, device, log_dir):
    batch_time = AverageMeter()
    losses_cle = AverageMeter()
    top1_cle = AverageMeter()
    top5_cle = AverageMeter()

    losses_adv = AverageMeter()
    top1_adv = AverageMeter()
    top5_adv = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        # input = input.cuda(async=True)

        target = target.to(device)
        input = input.to(device)

        # data_adv, targeted_labels = PGD(model, args, input, target, args.num_steps_1, args.step_size_1)
        data_adv = PGD(model, args, input, target, args.num_steps_1, args.step_size_1)
        with torch.no_grad():
            # compute output
            output_clean = model(input)
            output_adv = model(data_adv)
            loss_clean = criterion(output_clean, target)
            loss_adv = criterion(output_adv, target)

            # measure accuracy and record loss
            prec1_cle, prec5_cle = accuracy(output_clean.data, target, topk=(1, 5))
            losses_cle.update(loss_clean.item(), input.size(0))
            top1_cle.update(prec1_cle[0], input.size(0))
            top5_cle.update(prec5_cle[0], input.size(0))

            prec1_adv, prec5_adv = accuracy(output_adv.data, target, topk=(1, 5))
            losses_adv.update(loss_adv.item(), input.size(0))
            top1_adv.update(prec1_adv[0], input.size(0))
            top5_adv.update(prec5_adv[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0 and args.local_rank == 0:
                print('Test_clean: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses_cle,
                    top1=top1_cle, top5=top5_cle))

                print('Test_adv: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses_adv,
                    top1=top1_adv, top5=top5_adv))
    top1_cle_gather = [torch.ones_like(top1_cle.avg) for _ in range(torch.distributed.get_world_size())]
    top5_cle_gather = [torch.ones_like(top5_cle.avg) for _ in range(torch.distributed.get_world_size())]
    top1_adv_gather = [torch.ones_like(top1_adv.avg) for _ in range(torch.distributed.get_world_size())]
    top5_adv_gather = [torch.ones_like(top5_adv.avg) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(top1_cle_gather, top1_cle.avg, async_op=False)
    torch.distributed.all_gather(top5_cle_gather, top5_cle.avg, async_op=False)
    torch.distributed.all_gather(top1_adv_gather, top1_adv.avg, async_op=False)
    torch.distributed.all_gather(top5_adv_gather, top5_adv.avg, async_op=False)
    top1_cle_mean = torch.mean(torch.stack(top1_cle_gather))
    top5_cle_mean = torch.mean(torch.stack(top5_cle_gather))
    top1_adv_mean = torch.mean(torch.stack(top1_adv_gather))
    top5_adv_mean = torch.mean(torch.stack(top5_adv_gather))
    if args.local_rank == 0:
        print(' * Clean Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_cle_mean, top5_cle_mean))
        print(' * Adv Prec@1 {0:.3f} Prec@5 {1:.3f}'.format(top1_adv_mean, top5_adv_mean))
        print(' * Cl Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_cle, top5=top5_cle))
        print(' * Ad Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1_adv, top5=top5_adv))

    return top1_adv.avg, top5_adv.avg


if __name__ == '__main__':
    main()
