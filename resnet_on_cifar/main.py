from __future__ import print_function

import argparse
import os
import datetime
import sys
import time

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

from shb import SHB
from utils import get_git_diff, get_git_hash, Logger
from models import resnet20_cifar


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def loss_fn(args, output, target, model, device, reduction='mean'):
  if args.wd > 0:
    reg_loss = 0.0
    for name, weight in model.named_parameters():
      if 'weight' in name:
        reg_loss += torch.sum(weight ** 2)
    reg_loss *= 0.5 * args.wd
  else:
    reg_loss = 0.0
  if args.bd > 0:
    regb_loss = 0.0
    for name, bias in model.named_parameters():
      if 'bias' in name:
        regb_loss += torch.sum(bias ** 2)
    regb_loss *= 0.5 * args.bd
    reg_loss += regb_loss
  if args.loss == 'nll':
    loss = F.nll_loss(output, target, reduction=reduction) + reg_loss
  else:
    onehot_target = torch.zeros((target.shape[0], 10),
                                dtype=torch.float32, device=device)
    onehot_target.scatter_(1, target.view(-1, 1), 1)
    loss = F.mse_loss(torch.exp(output), onehot_target,
                      reduction=reduction) + reg_loss
  return loss


def train(args, model, device, train_loader, optimizer, step, writer):
  compute_time = AverageMeter()
  data_time = AverageMeter()
  losses = AverageMeter()
  accuracies = AverageMeter()

  model.train()

  end_tm = time.time()
  for batch_idx, (data, target) in enumerate(train_loader):
    data_time.update(time.time() - end_tm)

    optimizer.zero_grad()
    pl_data, pl_target = data.to(device), target.to(device)
    bs = pl_data.size(0)
    output = model(pl_data)
    pred = output.max(1, keepdim=True)[1]
    accuracy = pred.eq(pl_target.view_as(pred)).double().mean()
    loss = loss_fn(args, output, pl_target, model, device)
    loss.backward()
    optimizer.step()

    compute_time.update(time.time() - end_tm)
    losses.update(loss, bs)
    accuracies.update(accuracy, bs)

    if batch_idx % args.log_freq == 0:
      print(
        'Step: {0}\t'
        'Time {compute_time.val:.3f} ({compute_time.avg:.3f})\t'
        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        'Accuracy {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
          step, compute_time=compute_time, data_time=data_time,
          loss=losses, top1=accuracies,
        )
      )

      writer.add_scalar('metrics/batch/loss', loss, step)
      writer.add_scalar('metrics/avg/loss', losses.avg, step)
      writer.add_scalar('metrics/batch/accuracy', accuracy, step)
      writer.add_scalar('metrics/avg/accuracy', accuracies.avg, step)
      writer.add_scalar('metrics/batch/compute_time', compute_time.val, step)
      writer.add_scalar('metrics/avg/compute_time', compute_time.avg, step)
      writer.add_scalar('metrics/batch/data_time', data_time.val, step)
      writer.add_scalar('metrics/avg/data_time', data_time.avg, step)


    step += 1
    end_tm = time.time()

  for name, weight in model.named_parameters():
    writer.add_scalar('weights/{}/norm'.format(name), weight.norm(), step)
    writer.add_scalar('weights/{}/mean'.format(name), weight.mean(), step)
    writer.add_scalar('weights/{}/min'.format(name), weight.min(), step)
    writer.add_scalar('weights/{}/max'.format(name), weight.max(), step)

  return step


def eval_test(args, model, device, test_loader, step, writer):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss = loss_fn(args, output, target, model, device, 'sum')
      test_loss += loss.item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  data_size = len(test_loader.dataset)
  test_loss /= data_size
  accuracy = correct / data_size
  print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
    test_loss, 100.0 * accuracy,
  ))

  writer.add_scalar('metrics/accuracy', accuracy, step)
  writer.add_scalar('metrics/loss', test_loss, step)


def run_exp(args):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cuda':
    cudnn.benchmark = True
  print("Using: {}".format(device))

  batch_size = args.bs
  train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
      root=args.data_dir, train=True, download=True,
      transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.262)),
      ])),
    batch_size=batch_size, shuffle=True, num_workers=2,
  )

  test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
      root=args.data_dir, train=False, download=True,
      transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.243, 0.262)),
      ])),
    batch_size=batch_size, shuffle=False, num_workers=2,
  )

  model = resnet20_cifar().to(device)
  optimizer = SHB(model.parameters(), lr=args.lr, momentum=args.momentum)

  if args.checkpoint:
    if os.path.isfile(args.checkpoint):
      print('=> loading checkpoint "{}"'.format(args.checkpoint))
      checkpoint = torch.load(args.checkpoint)
      model.load_state_dict(checkpoint['state_dict'])
    else:
      print("=> no checkpoint found at '{}'".format(args.checkpoint))

  if args.drop_mode == 'freq':
    scheduler = torch.optim.lr_scheduler.StepLR(
      optimizer, args.drop_freq, args.drop_rate,
    )
  elif args.drop_mode == 'steps':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
      optimizer, args.drop_steps, args.drop_rate,
    )
  else:
    raise ValueError('Unknown drop_mode: {}'.format(args.drop_mode))
  step = 0
  train_writer = SummaryWriter(os.path.join(args.output_dir, 'logs', 'train'))
  test_writer = SummaryWriter(os.path.join(args.output_dir, 'logs', 'test'))

  for epoch in range(args.epochs):
    print("Epoch: {}".format(epoch))
    scheduler.step()
    step = train(args, model, device, train_loader, optimizer,
                 step, train_writer)
    eval_test(args, model, device, test_loader, step, test_writer)

    lr = optimizer.param_groups[0]['lr']
    m_coef = optimizer.param_groups[0]['momentum']
    train_writer.add_scalar('params/lr', lr, step)
    train_writer.add_scalar('params/m_coef', m_coef, step)
    train_writer.add_scalar('params/epoch', epoch + 1, step)
    if epoch % args.save_freq == 0:
      torch.save(
        {'epoch': epoch, 'step': step, 'state_dict': model.state_dict(), 
         'optimizer': optimizer.state_dict()},
        os.path.join(args.output_dir, 'models', 'checkpoint-{}.pt'.format(step)),
      )
  torch.save(
    {'epoch': epoch, 'step': step, 'state_dict': model.state_dict(), 
     'optimizer': optimizer.state_dict()},
    os.path.join(args.output_dir, 'models', 'checkpoint-{}.pt'.format(step)),
  )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTorch ResNet on CIFAR-10')
  parser.add_argument('--lr', default=0.1, type=float,
                      help='effective learning rate')
  parser.add_argument('--momentum', default=0.9, type=float,
                      help='effective momentum value')
  parser.add_argument('--loss', default='nll',
                      choices=['nll', 'mse'])
  parser.add_argument('--output_dir', type=str,
                      default=os.getenv('PT_OUTPUT_DIR', 'results'))
  parser.add_argument('--data_dir', dest='data_dir', type=str,
                      default=os.getenv('PT_DATA_DIR', 'data'))
  parser.add_argument('--bs', type=int, default=256)
  parser.add_argument('--log_freq', type=int, default=10)
  parser.add_argument('--save_freq', type=int, default=10, help="In epochs")
  parser.add_argument('--epochs', type=int, default=300)
  parser.add_argument('--drop_freq', type=int, default=10)
  parser.add_argument('--drop_steps', nargs='+', type=int,
                      default=[10, 30, 70, 150])
  parser.add_argument('--drop_rate', type=float, default=0.1)
  parser.add_argument('--drop_mode', choices=['steps', 'freq'], default='steps')
  parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
  parser.add_argument('--bd', type=float, default=0.0, help='bias decay')
  parser.add_argument('--checkpoint', type=str, default="", 
                      help="Checkpoint to restore the model from")
  args = parser.parse_args()

  logdir = args.output_dir
  os.makedirs(os.path.join(logdir, 'logs'), exist_ok=True)
  os.makedirs(os.path.join(logdir, 'models'), exist_ok=True)
  tm_suf = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

  with open(os.path.join(logdir, 'cmd-args_{}.log'.format(tm_suf)), 'w') as f:
    f.write(" ".join(sys.argv))
  with open(os.path.join(logdir, 'git-info_{}.log'.format(tm_suf)), 'w') as f:
    f.write('commit hash: {}'.format(get_git_hash()))
    f.write(get_git_diff())

  old_stdout = sys.stdout
  old_stderr = sys.stderr
  stdout_log = open(
    os.path.join(logdir, 'stdout_{}.log'.format(tm_suf)), 'a', 1,
  )
  stderr_log = open(
    os.path.join(logdir, 'stderr_{}.log'.format(tm_suf)), 'a', 1,
  )
  sys.stdout = Logger(sys.stdout, stdout_log)
  sys.stderr = Logger(sys.stderr, stderr_log)

  run_exp(args)

  sys.stdout = old_stdout
  sys.stderr = old_stderr
  stdout_log.close()
  stderr_log.close()
