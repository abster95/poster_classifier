import torch
from posters.dataset.torch_dataset import MoviePosters
from torch.utils.data import DataLoader
import torch.nn as nn
from posters.model.classifier import Classifier
import time

BATCH_SIZE = 8
NUM_WORKERS = 0
LR = 1e-3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

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

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target
        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target
        input_var = input.cuda()
        target_var = target.cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses))

if __name__ == "__main__":

    train_data = MoviePosters('train')
    val_data = MoviePosters('val')

    assert len(train_data.genres) == len(val_data.genres), \
        "The number of different genres shoud be the same in train and val"

    num_classes = len(train_data.genres)

    model = Classifier(num_classes=num_classes)
    model = model.cuda()

    train_loader = DataLoader(train_data,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=True, pin_memory=True)

    val_loader = DataLoader(val_data,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=False, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=LR)

    for epoch in range(20):
        train(train_loader, model, criterion, optimizer, epoch)
        validate(val_loader, model, criterion)