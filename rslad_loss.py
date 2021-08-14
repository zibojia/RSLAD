import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

def attack_pgd(model,train_batch_data,train_batch_labels,attack_iters=10,step_size=2/255.0,epsilon=8.0/255.0):
    ce_loss = torch.nn.CrossEntropyLoss().cuda()
    train_ifgsm_data = train_batch_data.detach() + torch.zeros_like(train_batch_data).uniform_(-epsilon,epsilon)
    train_ifgsm_data = torch.clamp(train_ifgsm_data,0,1)
    for i in range(attack_iters):
        train_ifgsm_data.requires_grad_()
        logits = model(train_ifgsm_data)
        loss = ce_loss(logits,train_batch_labels.cuda())
        loss.backward()
        train_grad = train_ifgsm_data.grad.detach()
        train_ifgsm_data = train_ifgsm_data + step_size*torch.sign(train_grad)
        train_ifgsm_data = torch.clamp(train_ifgsm_data.detach(),0,1)
        train_ifgsm_pert = train_ifgsm_data - train_batch_data
        train_ifgsm_pert = torch.clamp(train_ifgsm_pert,-epsilon,epsilon)
        train_ifgsm_data = train_batch_data + train_ifgsm_pert
        train_ifgsm_data = train_ifgsm_data.detach()
    return train_ifgsm_data

def rslad_inner_loss(model,
                teacher_logits,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=6.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False,reduce=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits
  
