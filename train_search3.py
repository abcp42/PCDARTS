import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import utils2
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--train_data_path', type=str, default='/content/data/train', help='location of the data corpus')
parser.add_argument('--val_data_path', type=str, default='/content/data/valid', help='location of the data corpus')
parser.add_argument('--test_data_path', type=str, default='/content/data/test', help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--image_size', type=int, default=64, help='batch size')
parser.add_argument('--n_class', type=int, default=3, help='number of classes')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.set=='cifar100':
    CIFAR_CLASSES = 100
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, args.n_class, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  
  _, _, n_classes, train_data,val_dat,test_dat = utils2.get_data(
        "custom", args.train_data_path,args.val_data_path,args.test_data_path, cutout_length=0, validation=True,validation2 = True,n_class = args.n_class, image_size = args.image_size)
  
  #balanced split to train/validation
  print(train_data)

  # split data to train/validation
  num_train = len(train_data)
  n_val = len(val_dat)
  n_test = len(test_dat)
  indices1 = list(range(num_train))
  indices2 = list(range(n_val))
  indices3 = list(range(n_test))
  train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices1)
  valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices2)
  test_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices3)


  train_queue = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           sampler=train_sampler,
                                           num_workers=2,
                                           pin_memory=True)
  valid_queue = torch.utils.data.DataLoader(val_dat,
                                           batch_size=args.batch_size,
                                           sampler=valid_sampler,
                                           num_workers=2,
                                           pin_memory=True)
  test_queue = torch.utils.data.DataLoader(test_dat,
                                           batch_size=args.batch_size,
                                           sampler=test_sampler,
                                           num_workers=2,
                                           pin_memory=True)
     
  """
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  """
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  bestMetric = -999
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    #if args.epochs-epoch<=1:
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)
    
    test_acc,test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)
    
    utils.save(model, os.path.join(args.save, 'weights.pt'))
    if(valid_acc > bestMetric):
        bestMetric = valid_acc
        utils.save(model, os.path.join(args.save, 'best_weights.pt'))
        

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    #try:
    #  input_search, target_search = next(valid_queue_iter)
    #except:
    #  valid_queue_iter = iter(valid_queue)
    #  input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  preds = np.asarray([])
  targets = np.asarray([])

  for step, (input, target) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(async=True)
    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    #objs.update(loss.data[0], n)
    #top1.update(prec1.data[0], n)
    #top5.update(prec5.data[0], n)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    
    #minha alteracao
    output = logits
    topk = (1,3)
    maxk = max(topk)
    batch_size = target.size(0)
    _, predicted = torch.max(output.data, 1)
    #minha alteracao
    preds = np.concatenate((preds,predicted.cpu().numpy().ravel()))
    #targets = np.concatenate((targets,target.cpu().numpy().ravel()))
    targets = np.concatenate((targets,target.data.cpu().numpy().ravel()))


    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  
  print(preds.shape)
  print(targets.shape)
  print('np.unique(targets):',np.unique(targets))
  print('np.unique(preds): ',np.unique(preds))
  from sklearn.metrics import classification_report
  from sklearn.metrics import accuracy_score
  print(accuracy_score(targets, preds))
  cr = classification_report(targets, preds,output_dict= True)
  a1,a2,a3 = cr['macro avg']['f1-score'] ,cr['macro avg']['precision'],cr['macro avg']['recall'] 
  topover = (a1+a2+a3)/3 
  print(classification_report(targets, preds))
  from sklearn.metrics import balanced_accuracy_score
  from sklearn.metrics import accuracy_score
  print(balanced_accuracy_score(targets, preds))
  print(accuracy_score(targets, preds))
  from sklearn.metrics import confusion_matrix
  matrix = confusion_matrix(targets, preds)
  print(matrix.diagonal()/matrix.sum(axis=1))
  print(matrix)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 
