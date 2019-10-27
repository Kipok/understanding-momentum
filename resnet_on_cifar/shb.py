import torch
from torch.optim import Optimizer


class SHB(Optimizer):
  def __init__(self, params, lr, momentum=0, weight_decay=0):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
      raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
    super(SHB, self).__init__(params, defaults)

  def __setstate__(self, state):
    super(SHB, self).__setstate__(state)

  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      momentum = group['momentum']

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data
        if weight_decay != 0:
          d_p.add_(weight_decay, p.data)
        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            buf.mul_(momentum).add_(1 - momentum, d_p)
          else:
            buf = param_state['momentum_buffer']
            buf.mul_(momentum).add_(1 - momentum, d_p)

          d_p = buf

        p.data.add_(-group['lr'], d_p)

    return loss
