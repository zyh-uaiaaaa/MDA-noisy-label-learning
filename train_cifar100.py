# -*- coding:utf-8 -*-
import os
import torch
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100

import argparse, sys
import datetime
import random
import numpy as np


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from model.cnn import MLPNet,CNN
import numpy as np
from common.utils import accuracy



global update

import torch.nn as nn


device = torch.device('cuda:0')


def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


def loss_t(y_1, y_2, t, ind, co_lambda, total_loss):
    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()
    total_loss[ind] = loss_pick
    return total_loss
    

def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1, update=None):
    

    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) 
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) 
    loss_pick = (loss_pick_1 + loss_pick_2 ).cpu()
    
    if update is None:
        ind_sorted = np.argsort(loss_pick.data)
        loss_sorted = loss_pick[ind_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_sorted))

        pure_ratio = np.sum(noise_or_not[ind[ind_sorted[:num_remember]]])/float(num_remember)

        ind_update=ind_sorted[:num_remember]

        # exchange
        loss = torch.mean(loss_pick[ind_update])
        return loss, loss, pure_ratio, pure_ratio
    else:
        ind_update = []
        for i in range(y_1.size()[0]):
            if ind[i] in update:
                ind_update.append(i)
        ind_update = np.array(ind_update)
        loss = torch.mean(loss_pick[ind_update])
        return loss, loss, None, None

    




class JoCoR:
    def __init__(self, args, train_dataset, device, input_channel, num_classes, whole):

        # Hyper Parameters
        self.batch_size = 512
        learning_rate = args.lr

        if args.noise_type == "asymmetric":
            forget_rate = args.noise_rate / 2
        else:
            forget_rate = args.noise_rate
        

        self.noise_or_not = train_dataset.noise_or_not

        # Adjust learning rate and betas for Adam Optimizer
        mom1 = 0.9
        mom2 = 0.9
        self.alpha_plan = [learning_rate] * args.n_epoch
        self.beta1_plan = [mom1] * args.n_epoch

        for i in range(args.epoch_decay_start, args.n_epoch):
            self.alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * learning_rate
            self.beta1_plan[i] = mom2

        # define drop rate schedule
        self.rate_schedule = np.ones(args.n_epoch) * forget_rate
        self.rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate ** args.exponent, args.num_gradual)

        self.device = device
        self.num_iter_per_epoch = args.num_iter_per_epoch
        self.print_freq = args.print_freq
        self.co_lambda = args.co_lambda
        self.n_epoch = args.n_epoch
        self.train_dataset = train_dataset


        self.model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
        self.model1 = torch.nn.DataParallel(self.model1)
        self.model2 = torch.nn.DataParallel(self.model2)



        self.model1.to(device)
        self.model2.to(device)
 

        self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=learning_rate)
        
        
        self.loss_fn = loss_jocor
        self.loss_t = loss_t
        self.whole = whole

        self.adjust_lr = args.adjust_lr
        self.flip = args.flip
        
    def evaluate(self, test_loader):
        print('Evaluating ...')
        self.model1.eval()  # Change model to 'eval' mode.
        self.model2.eval()  # Change model to 'eval' mode

        correct1 = 0
        total1 = 0
        for images, _, labels, _ in test_loader:
            images = Variable(images).to(self.device)
            logits1 = self.model1(images)
            logits2 = self.model2(images)
            logits = logits1 + logits2
            outputs1 = F.softmax(logits, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

        

        acc1 = 100 * float(correct1) / float(total1)
        
        return acc1
    
    # Train the Model
    def train(self, train_loader, epoch, update):
        if self.whole:  
            if epoch < 100:
                with torch.no_grad():
                    self.model1.eval()
                    self.model2.eval()
                    total_loss = torch.zeros(train_loader.dataset.__len__())
                    for i, (images, imagesf, labels, indexes) in enumerate(train_loader):
                        ind = indexes.cpu().numpy().transpose()
                        labels = Variable(labels).to(self.device)                    
                        imagesf = Variable(imagesf).to(self.device)
                        logits1 = self.model1(imagesf)
                        logits2 = self.model2(imagesf)                    
                        t_loss = self.loss_t(logits1, logits2, labels, ind, self.co_lambda, total_loss)


                        total_loss = t_loss


                    ind_sorted = np.argsort(total_loss.data)
                    remember_rate = 1 - self.rate_schedule[epoch]
                    num_remember = int(remember_rate * train_loader.dataset.__len__())
                    update = ind_sorted[:num_remember]
        else:
            update = None           
                
                
        

        print('Training ...')
        self.model1.train()  # Change model to 'train' mode.
        self.model2.train()  # Change model to 'train' mode


        self.adjust_learning_rate(self.optimizer, epoch)

        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        running_loss = 0
        iter_cnt = 0
        pure_ratio_1_list = []
        pure_ratio_2_list = []

        for i, (images, imagesf, labels, indexes) in enumerate(train_loader):
            ind = indexes.cpu().numpy().transpose()
            

            images = Variable(images).to(self.device)
            imagesf = Variable(imagesf).to(self.device)
            labels = Variable(labels).to(self.device)

            # Forward + Backward + Optimize
            logits1 = self.model1(images)
            prec1 = accuracy(logits1, labels, topk=(1,))
            train_total += 1
            train_correct += prec1
            
            logits2 = self.model2(images)
            prec2 = accuracy(logits2, labels, topk=(1,))
            train_total2 += 1
            train_correct2 += prec2

            loss_1, loss_2, pure_ratio_1, pure_ratio_2 = self.loss_fn(logits1, logits2, labels, self.rate_schedule[epoch], ind, self.noise_or_not, self.co_lambda, update=update)
            
            logits = (logits1 + logits2)/2
            ncl_loss = torch.nn.MSELoss()(logits1, logits) + torch.nn.MSELoss()(logits2, logits)
            
            loss = loss_1 + ncl_loss.cpu()
            iter_cnt += 1
            running_loss += loss_1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss = running_loss/iter_cnt

        train_acc1 = float(train_correct) / float(train_total)
        train_acc2 = float(train_correct2) / float(train_total2)
        return train_acc1, train_acc2, update
            


    def adjust_learning_rate(self, optimizer, epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)  # Only change beta1



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.8)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=0.8)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=100)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--co_lambda', type=float, default=0.65)
parser.add_argument('--lam', type=float, default=1.0)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--flip', type=bool, help='flip images?', default=False)
parser.add_argument('--whole', type=bool, help='use my method?', default=True)



args = parser.parse_args()

seed = args.seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# Hyper Parameters
batch_size = 512
learning_rate = args.lr


input_channel = 3
num_classes = 100
init_epoch = 5
args.epoch_decay_start = 100
args.model_type = "cnn"
train_dataset = CIFAR100(root='data/',
                        download=True,
                        train=True,
                        transform=transforms.ToTensor(),
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate)

test_dataset = CIFAR100(root='data/',
                       download=True,
                       train=False,
                       transform=transforms.ToTensor(),
                       noise_type=args.noise_type,
                       noise_rate=args.noise_rate)



if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate




update = None
# Data Loader (Input Pipeline)
print('loading dataset...')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=args.num_workers,
                                           drop_last=True,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=args.num_workers,
                                          drop_last=True,
                                          shuffle=False)
# Define models
print('building model...')

model = JoCoR(args, train_dataset, device, input_channel, num_classes, whole=args.whole)

epoch = 0
train_acc1 = 0
train_acc2 = 0


acc_list = []
# training
for epoch in range(1, args.n_epoch):
    train_acc1, train_acc2, update = model.train(train_loader, epoch, update)
    
    test_acc1 = model.evaluate(test_loader)

    
    if epoch >= 190:
        acc_list.extend([test_acc1])

avg_acc = sum(acc_list)/len(acc_list)
print("the average acc in last 10 epochs: {}".format(str(avg_acc)))




