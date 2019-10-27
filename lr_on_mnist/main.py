from __future__ import print_function

import argparse
import os
import datetime
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms

from qhm import QHM
from utils import get_git_diff, get_git_hash, Logger


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(784, 10)

  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.fc1(x)
    return F.log_softmax(x, dim=1)


def loss_fn(args, output, target, model, device, reduction='mean'):
  if args.wd > 0:
    reg_loss = 0.5 * args.wd * torch.sum(model.fc1.weight ** 2)
  else:
    reg_loss = 0.0
  if args.bd > 0:
    reg_loss += 0.5 * args.bd * torch.sum(model.fc1.bias ** 2)
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
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    def closure():
      optimizer.zero_grad()
      pl_data, pl_target = data.to(device), target.to(device)
      output = model(pl_data)
      loss = loss_fn(args, output, pl_target, model, device)
      loss.backward()
      return loss

    loss = optimizer.step(closure)
    if batch_idx % 10 == 0:
      writer.add_scalar('metrics/batch_loss', loss, step)
    step += 1
  return step


def eval_train(args, model, device, train_loader, step, writer):
  model.eval()
  train_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in train_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      loss = loss_fn(args, output, target, model, device, 'sum')
      train_loss += loss.item()
      pred = output.max(1, keepdim=True)[1]
      correct += pred.eq(target.view_as(pred)).sum().item()

  data_size = len(train_loader.dataset)
  train_loss /= data_size
  accuracy = correct / data_size
  print('\nTrain set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
    train_loss, 100.0 * accuracy,
  ))

  writer.add_scalar('metrics/accuracy', accuracy, step)
  writer.add_scalar('metrics/loss', train_loss, step)

  writer.add_scalar('weights/w_norm', model.fc1.weight.norm().item(), step)
  writer.add_scalar('weights/w_mean', model.fc1.weight.mean().item(), step)
  writer.add_scalar('weights/w_min', model.fc1.weight.min().item(), step)
  writer.add_scalar('weights/w_max', model.fc1.weight.max().item(), step)

  writer.add_scalar('weights/b_norm', model.fc1.bias.norm().item(), step)
  writer.add_scalar('weights/b_mean', model.fc1.bias.mean().item(), step)
  writer.add_scalar('weights/b_min', model.fc1.bias.min().item(), step)
  writer.add_scalar('weights/b_max', model.fc1.bias.max().item(), step)


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
  use_cuda = torch.cuda.is_available()
  print("Use cuda: ", use_cuda)

  device = torch.device("cuda" if use_cuda else "cpu")

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  batch_size = args.bs
  train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
      args.data_dir, train=True, download=True,
      transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])
    ),
    batch_size=batch_size, shuffle=True, **kwargs,
  )

  test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
      args.data_dir, train=False,
      transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
      ])
    ),
    batch_size=batch_size, shuffle=False, **kwargs,
  )

  model = Net().to(device)
  optimizer = QHM(model.parameters(), lr=args.lr, momentum=args.momentum, nu=args.nu)
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

  for epoch in range(1, args.epochs + 1):
    print("Epoch: {}".format(epoch))
    scheduler.step()
    step = train(args, model, device, train_loader, optimizer, step, train_writer)
    eval_test(args, model, device, test_loader, step, test_writer)
    eval_train(args, model, device, train_loader, step, train_writer)

    lr = optimizer.param_groups[0]['lr']
    m_coef = optimizer.param_groups[0]['momentum']
    train_writer.add_scalar('params/lr', lr, step)
    train_writer.add_scalar('params/m_coef', m_coef, step)
    train_writer.add_scalar('params/epoch', epoch, step)
 #   torch.save(model.state_dict(), os.path.join(args.output_dir, 'models',
 #                                               'checkpoint-{}.pt'.format(step)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description='PyTorch MNIST logistic regression',
  )
  parser.add_argument('--lr', default=0.1, type=float,
                      help='effective learning rate')
  parser.add_argument('--momentum', default=0.9, type=float,
                      help='effective momentum value')
  parser.add_argument('--nu', default=1.0, type=float,
                      help='effective nu value')
  parser.add_argument('--loss', default='nll',
                      choices=['nll', 'mse'])
  parser.add_argument('--output_dir', type=str,
                      default=os.getenv('PT_OUTPUT_DIR', 'results'))
  parser.add_argument('--data_dir', dest='data_dir', type=str,
                      default=os.getenv('PT_DATA_DIR', 'data'))
  parser.add_argument('--bs', type=int, default=128)
  parser.add_argument('--epochs', type=int, default=300)
  parser.add_argument('--drop_freq', type=int, default=10)
  parser.add_argument('--drop_steps', nargs='+', type=int,
                      default=[10, 30, 70, 150])
  parser.add_argument('--drop_rate', type=float, default=0.5)
  parser.add_argument('--drop_mode', choices=['steps', 'freq'], default='steps')
  parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
  parser.add_argument('--bd', type=float, default=0.0, help='bias decay')
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
