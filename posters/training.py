import torch
from posters.dataset.torch_dataset import MoviePosters
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet50, resnet101
import torch.nn as nn
from posters.model.classifier import Classifier
import time

BATCH_SIZE = 32
NUM_WORKERS = 10
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

def accuracy(output, target, thresh=0.7):
    """Computes the precision@k for the specified values of k"""
    correct = 0.0
    fp = 0.0
    fn = 0.0
    total = 0.0
    with torch.no_grad():
        thresholded = (output > thresh).float()
        total += target.size(0)
        num_classes = target.size(1)
        correct += ((thresholded==target).sum(axis=-1).float() / num_classes).sum().item()
        fp += ((thresholded==(1-target)).sum(axis=-1).float() / num_classes).sum().item()
        fn += (((1-thresholded)==target).sum(axis=-1).float() / num_classes).sum().item()
    return correct/total, fp/total, fn/total

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

        accurate, fp, fn = accuracy(output, target_var)

        print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc:.4f}\t'
                'FP {fp:.4f}\t'
                'FN {fn:.4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=accurate,
                fp=fp, fn=fn))

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

        accurate, fp, fn = accuracy(output, target_var)
        print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {acc:.4f}\t'
                'FP {fp:.4f}\t'
                'FN {fn:.4f}'.format(
                i, len(val_loader), batch_time=batch_time,
                loss=losses, acc=accurate,
                fp=fp, fn=fn))

if __name__ == "__main__":

    train_data = MoviePosters('train')
    val_data = MoviePosters('val')

    assert len(train_data.genres) == len(val_data.genres), \
        "The number of different genres shoud be the same in train and val"

    num_classes = len(train_data.genres)
    model = Classifier(backbone=resnet50, num_classes=num_classes)
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
