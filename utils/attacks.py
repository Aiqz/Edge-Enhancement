#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._jit_internal import weak_script_method
import cv2
from torch.autograd import Variable


# Projected Gradient Descent
def PGD(model, args, inputs, targets, num_steps, step_size):
    x = inputs.detach()

    if args.random:
        x = x + torch.zeros_like(x).uniform_(-args.epsilon, args.epsilon)
        x = torch.clamp(x, 0, 1)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, targets, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0, 1)  # data sclale

    return x


# A targeted_PGD white-box attacker with random target label.
def targeted_PGD(model, args, inputs, labels, num_steps, step_size, nclass, device):

    x = inputs.detach()
    # we consider targeted attacks when evaluating under the white-box settings,
    # where the targeted class is selected uniformly at random
    label_offset = torch.randint(low=1, high=nclass, size=labels.shape).to(device)
    # print('label_offset:{}'.format(label_offset))
    target_labels = torch.fmod(labels + label_offset, nclass)

    if args.random:
        x = x + torch.zeros_like(x).uniform_(-args.epsilon, args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() - step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)  # data sclale

    return x, target_labels


def targeted_PGD_trick(model, args, inputs, labels, num_steps, step_size, nclass, device):

    x = inputs.detach()
    # we consider targeted attacks when evaluating under the white-box settings,
    # where the targeted class is selected uniformly at random
    label_offset = torch.randint(low=1, high=nclass, size=labels.shape).to(device)
    # print('label_offset:{}'.format(label_offset))
    target_labels = torch.fmod(labels + label_offset, nclass)

    if args.random:
        init_start = torch.Tensor(x.shape).uniform_(-args.epsilon, args.epsilon).to(device)
        start_from_noise_index = torch.gt(torch.rand([]), args.prob_start_from_clean).type(torch.float32).to(device)
        x = x + start_from_noise_index * init_start

        x = torch.clamp(x, 0.0, 1.0)

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        # x = x.detach() + step_size * torch.sign(grad.detach())  ###wrong
        x = x.detach() - step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)  # data sclale

    return x, target_labels


class LabelSmoothLoss(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


# Compute loss for trick model
def compute_loss_and_error(logits, label, label_smoothing=0.):
    loss_function = LabelSmoothLoss(label_smoothing)
    loss = loss_function(logits, label.long())
    return loss


# FGSM
def FGSM(model, inputs, target, targeted=False, step_size= 0.007):

    x = inputs.detach()
    x.requires_grad_()

    with torch.enable_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, target, reduction='sum')

    grad = torch.autograd.grad(loss, [x])[0]

    if targeted:
        x = x.detach() - step_size * torch.sign(grad.detach())
    else:
        x = x.detach() + step_size * torch.sign(grad.detach())

    x = torch.clamp(x, 0.0, 1.0)  # data sclale

    return x


def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]


# CW with Linf norm
def CWLinfAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', n_class=10, cur_device=None):
    
    model.eval()
    device = cur_device
    x = x.to(device)
    y = y.to(device)
    # print(x.shape)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    target = target[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    target = target if len(target.shape) == 1 else target.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    
    n_class = n_class
    one_hot_y = torch.zeros(y.size(0), n_class).to(device)
    one_hot_y[torch.arange(y.size(0)), y] = 1

    # random_start
    x.requires_grad = True 
    if isinstance(magnitude, Variable):
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude.item(), magnitude.item())
    else:
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.to(device)
    # print(x.shape)
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    if previous_p is not None:
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    with torch.enable_grad():
        for _iter in range(max_iters):
            
            outputs = model(adv_imgs)

            correct_logit = torch.sum(one_hot_y * outputs, dim=1)
            if target is not None:
                # print(target.shape)
                wrong_logit = torch.zeros(target.size(0), n_class).to(device)
                wrong_logit[torch.arange(target.size(0)), target] = 1
                # print(wrong_logit.shape)
                # print(outputs.shape)
                wrong_logit = torch.sum(wrong_logit * outputs, dim=1)
            else:
                wrong_logit,_ = torch.max((1-one_hot_y) * outputs-1e4*one_hot_y, dim=1)

            loss = -torch.sum(F.relu(correct_logit-wrong_logit+50))

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            # step_size: 0.00392 for tinyimagenet; 0.01 for mnist
            step_size = 0.00392
            adv_imgs.data += step_size * torch.sign(grads.data) 

            # the adversaries' pixel value should within max_x and min_x due 
            # to the l_infinity / l2 restriction

            adv_imgs = torch.max(torch.min(adv_imgs, x + magnitude), x - magnitude)

            adv_imgs.clamp_(0, 1)

            adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)

    adv_imgs.clamp_(0, 1)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p


# ALP
class ALP:
    def __init__(self, step_size=0.003, epsilon=0.047, perturb_steps=5, beta=1.0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta

    def reset_steps(self, k):
        self.perturb_steps = k

    @weak_script_method
    def PGD_Linf(self, model, x_natural, y):

        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def loss(self, model, logits, logits_adv, y, optimizer):
        model.train()
        optimizer.zero_grad()

        loss_robust = 0.5 * F.cross_entropy(logits, y) + 0.5 * F.cross_entropy(logits_adv, y)
        loss_alp = F.mse_loss(logits, logits_adv)
        loss = loss_robust + self.beta * loss_alp

        return loss


# Targeted ALP for Tiny ImageNet
class targeted_ALP:
    def __init__(self, step_size=0.003, epsilon=0.047, perturb_steps=5, beta=1.0, n_class = 200):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.n_class = n_class

    def reset_steps(self, k):
        self.perturb_steps = k

    @weak_script_method
    def PGD_Linf(self, model, x_natural, y):

        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss_c, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def tarPGD_Linf(self, model, x_natural, y, device):

        model.eval()
        label_offset = torch.randint(low=1, high=self.n_class, size=y.shape).to(device)
        target_labels = torch.fmod(y + label_offset, self.n_class)

        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_c = F.cross_entropy(model(x_adv), target_labels)
            grad = torch.autograd.grad(loss_c, [x_adv])[0].detach()
            x_adv = x_adv.detach() - self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def loss(self, model, logits, logits_adv, y, optimizer):
        model.train()
        optimizer.zero_grad()

        loss_robust = 0.5 * F.cross_entropy(logits, y) + 0.5 * F.cross_entropy(logits_adv, y)
        loss_alp = F.mse_loss(logits, logits_adv)
        loss = loss_robust + self.beta * loss_alp

        return loss


# Targeted ALP for ImageNet
def tar_alp_imagenet(model, args, inputs, labels, num_steps, step_size, device):

    x = inputs.detach()

    label_offset = torch.randint(low=1, high=1000, size=labels.shape).to(device)
    target_labels = torch.fmod(labels + label_offset, 1000)

    x = x + 0.001 * torch.randn(x.shape).to(device).detach()

    for i in range(num_steps):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target_labels, reduction='sum')
        grad = torch.autograd.grad(loss, [x])[0]
        # x = x.detach() + step_size * torch.sign(grad.detach())  ###wrong
        x = x.detach() - step_size * torch.sign(grad.detach())
        x = torch.min(torch.max(x, inputs - args.epsilon), inputs + args.epsilon)
        x = torch.clamp(x, 0.0, 1.0)  # data sclale

    return x, target_labels


def squared_l2_norm(x):
    flattened = x.view(x.shape[0], -1)
    return (flattened ** 2).mean(1)
    

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

# TRADES
class Trades:
    def __init__(self, step_size=0.003, epsilon=0.047, perturb_steps=5, beta=1.0):
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.beta = beta
        self.criterion_kl = nn.KLDivLoss(reduction="batchmean")

    def reset_steps(self, k):
        self.perturb_steps = k

    @weak_script_method
    def PGD_L2(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            with torch.enable_grad():
                x_adv.requires_grad_()
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            grad /= l2_norm(grad).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1e-8
            x_adv = x_adv.detach() + self.step_size * grad

            delta = x_adv - x_natural
            delta_norm = l2_norm(delta)
            cond = delta_norm > self.epsilon
            delta[cond] *= self.epsilon / delta_norm[cond].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x_adv = x_natural + delta
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def PGD_Linf(self, model, x_natural, logits):
        model.eval()
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device='cuda').detach()
        prob = F.softmax(logits, dim=-1)

        for _ in range(self.perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0].detach()
            x_adv = x_adv.detach() + self.step_size * torch.sign(grad)
            x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        return x_adv

    @weak_script_method
    def loss(self, model, logits, x_adv, labels, optimizer):
        model.train()
        optimizer.zero_grad()
        prob = F.softmax(logits, dim=-1)
        loss_natural = F.cross_entropy(logits, labels)
        loss_robust = self.criterion_kl(F.log_softmax(model(x_adv), dim=1), prob)
        loss = loss_natural + self.beta * loss_robust

        return loss


# AVmixup
class AVmixup:
    def __init__(self, args, gamma, lambda1, lambda2, step_size, num_steps, num_classes=200, device='cuda'):
        self.args = args
        self.gamma = gamma
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.device = device

    def _label_smoothing(self, one_hot, factor):
        return one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(self.num_classes - 1))

    def perturb(self, model, inputs, targets):
        """
            Given a set of examples (inputs, targets), returns a set of adversarial
            examples within epsilon of inputs in l_infinity norm.
        """
        x = inputs.detach()

        if self.args.random:
            x = x + torch.zeros_like(x).uniform_(-self.args.epsilon, self.args.epsilon)
            x = torch.clamp(x, 0, 1)

        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)
                log_prob = F.log_softmax(logits, dim=1)
                loss = -torch.sum(log_prob * targets)
                # loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.args.epsilon), inputs + self.args.epsilon)
            x = torch.clamp(x, 0, 1)  # data sclale
        perturb = (x - inputs) * self.gamma
        adversarial_vertex = inputs + perturb
        adversarial_vertex = torch.clamp(adversarial_vertex, 0, 1)
        y_nat = self._label_smoothing(targets, self.lambda1)
        y_vertex = self._label_smoothing(targets, self.lambda2)
        x_weight = np.random.beta(1.0, 1.0, [x.shape[0], 1, 1, 1])
        x_weight_torch = torch.from_numpy(x_weight).to(self.device)
        y_weight = torch.from_numpy(np.reshape(x_weight, [-1, 1])).to(self.device)
        x = inputs * x_weight_torch + adversarial_vertex * (1 - x_weight_torch)
        y = y_nat * y_weight + y_vertex * (1 - y_weight)
        return x.to(torch.float), y

    def tar_perturb(self, model, inputs, targets):
        """
            Given a set of examples (inputs, targets), returns a set of adversarial
            examples within epsilon of inputs in l_infinity norm.
        """
        x = inputs.detach()

        label_offset = torch.randint(low=1, high=self.num_classes, size=targets.shape).to(self.device)
        target_labels = torch.fmod(targets + label_offset, self.num_classes)


        if self.args.random:
            x = x + torch.zeros_like(x).uniform_(-self.args.epsilon, self.args.epsilon)
            x = torch.clamp(x, 0, 1)

        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = model(x)
                log_prob = F.log_softmax(logits, dim=1)
                loss = -torch.sum(log_prob * target_labels)
                # loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            # x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = x.detach() - self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.args.epsilon), inputs + self.args.epsilon)
            x = torch.clamp(x, 0, 1)  # data sclale
        perturb = (x - inputs) * self.gamma
        adversarial_vertex = inputs + perturb
        adversarial_vertex = torch.clamp(adversarial_vertex, 0, 1)
        y_nat = self._label_smoothing(targets, self.lambda1)
        y_vertex = self._label_smoothing(targets, self.lambda2)
        x_weight = np.random.beta(1.0, 1.0, [x.shape[0], 1, 1, 1])
        x_weight_torch = torch.from_numpy(x_weight).to(self.device)
        y_weight = torch.from_numpy(np.reshape(x_weight, [-1, 1])).to(self.device)
        x = inputs * x_weight_torch + adversarial_vertex * (1 - x_weight_torch)
        y = y_nat * y_weight + y_vertex * (1 - y_weight)
        return x.to(torch.float), y


