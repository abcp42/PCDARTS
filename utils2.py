""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc


def get_data(dataset, data_path,val1_data_path,val2_data_path, cutout_length, validation,validation2 = False,n_class = 3,image_size = 64):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif dataset == 'custom':
        dset_cls = dset.ImageFolder
        n_classes = n_class #2 to mama
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length,image_size)
    if dataset == 'custom':
        print("DATA PATH:", data_path)
        trn_data = dset_cls(root=data_path, transform=trn_transform)
        #dataset_loader = torch.utils.data.DataLoader(trn_data,
        #                                     batch_size=16, shuffle=True,
        #                                     num_workers=1)
        
    else:
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    if dataset == 'custom':
        shape = [1, image_size, image_size,3]
    else:
        shape = trn_data.train_data.shape
    print(shape)
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]
    print('input_size: uitls',input_size)

    ret = [input_size, input_channels, n_classes, trn_data]
        
    if validation: # append validation data
        if dataset == 'custom':
            dset_cls = dset.ImageFolder(val1_data_path,transform=val_transform)
            ret.append(dset_cls)
        else:
            ret.append(dset_cls(root=data_path, train=False, download=True, transform=val_transform))
    if validation2:
        if dataset == 'custom':
            dset_cls = dset.ImageFolder(val2_data_path,transform=trn_transform)
            ret.append(dset_cls)
    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    #print('output:',output)
    #print('target:',target)
    #print('maxk:',maxk)
    ###TOP 5 NAO EXISTE NAS MAAMAS OU NO GEO. TEM QUE TRATAR
    maxk = 3 # Ignorando completamente o top5

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, ckpt_dir, is_best=False, is_best_overall =False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'w_optimizer_state_dict': w_optimizer.state_dict(),
            'a_optimizer_state_dict': a_optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
    if is_best_overall:
        best_filename = os.path.join(ckpt_dir, 'best_overall.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, filename='checkpoint.pth.tar'):
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer_state_dict'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,w_optimizer,a_optimizer,loss



def save_checkpoint2(model,epoch,optimizer,loss, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'model.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint2(model,epoch,optimizer,loss, filename='model.pth.tar'):
    filename=filename+'checkpoint.pth.tar'
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,optimizer,loss
