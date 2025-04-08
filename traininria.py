import argparse
import numpy as np
import random
import os
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from torch.autograd import Variable
from torch.utils.data import DataLoader
from LQYDataLoaderberlin import get_training_set,get_test_set
import torchvision
import functools
from PIL import Image, ImageStat
import time
from unet_model import UNet

#from tiramisu_de import FCDenseNet67
#from efficientunet import *
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"
key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
}
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
def map01(tensor):
    #input/output:tensor
    maxa=np.copy(tensor.numpy())
    mina=np.copy(tensor.numpy())
    maxa[:,0,:,:]=255.0
    maxa[:,1,:,:]=255.0
    maxa[:,2,:,:]=255.0
    mina[:,0,:,:]=0.0
    mina[:,1,:,:]=0.0
    mina[:,2,:,:]=0.0
    return torch.from_numpy( (tensor.numpy() - mina) / (maxa-mina) )
    
def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()
    imPred += 1
    imLab += 1
    imPred = imPred * (imLab>0)
    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
    intersection, bins=numClass, range=(1, numClass))
    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)

def train(epoch,args,model,training_data_loader,optimizer):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input = Variable(batch[0])
        #print(input)
        #input=map01(input)
        target = Variable(batch[1])
        #print(input)
        target =target.squeeze(1)
        criterion = nn.BCELoss() 
        criterion2 = nn.NLLLoss2d() 
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
            criterion = criterion.cuda()
            criterion2 = criterion2.cuda()
        model.train()
        
        input2 = model(input)
        hp=0.01        
        loss = hp*criterion(input2[0], target.float())+hp*criterion(input2[1], target.float())+hp*criterion(input2[2], target.float())+hp*criterion(input2[3], target.float())+hp*criterion(input2[4], target.float())+criterion2(input2[5], target.long())
        print(loss)
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
    train_loss=epoch_loss / len(training_data_loader)
    return train_loss

def test(args,model,testing_data_loader,optimizer):
    totalloss = 0
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    n=1
    for batch in testing_data_loader:
        input = Variable(batch[0],volatile=True)
        #input=map01(input)
        target = Variable(batch[1],volatile=True)
        imgout3 = input.data
        target =target.squeeze(1)
        npimgra=target.squeeze(0)
        npimgr=npimgra.cpu().numpy()
        criterion = nn.BCELoss() 
        criterion2 = nn.NLLLoss2d()
         
        if args.cuda:
            input = input.cuda()
            target = target.cuda()
            criterion = criterion.cuda()
            criterion2 = criterion2.cuda()
        optimizer.zero_grad()
        model.eval()
        predictiona = model(input)
        prediction=predictiona[5]
        loss = criterion2(prediction, target.long())
        totalloss += loss.data
        imgout = prediction.data
        imgout1=imgout.squeeze(0)      
        values,imgout1a=imgout1.max(0) 
        npimg = imgout1a.cpu().numpy()
        n=n+1

        acc, pix = accuracy(npimg, npimgr)
        intersection, union = intersectionAndUnion(npimg, npimgr,2)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
    avg_test_loss=totalloss / len(testing_data_loader)
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    return avg_test_loss,iou,acc_meter


def checkpoint(epoch,args,model):
    model_out_path = args.checkpoint+'/best_model.pth'
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    if args.cuda:
      torch.cuda.manual_seed(args.seed)
    print('===> Loading datasets')

#loading training dataset
    train_set = get_training_set(args.root_dataset,args.img_size, target_mode= args.target_mode, colordim=args.colordim)
    training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.trainbatchsize, shuffle=True)
#loading validation dataset
    test_set = get_test_set(args.root_dataset,args.img_size, target_mode= args.target_mode, colordim=args.colordim)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.validationbatchsize, shuffle=False)
    num_class=args.num_class
    #model=get_efficientunet_b0(out_channels=num_class, concat_input=True, pretrained=False,in_channels=args.colordim)
    #model = torch.nn.DataParallel(model, device_ids=[0,1])
    model=UNet(n_channels=args.colordim,n_classes=num_class)
    #model=DeepLab(backbone='xception',num_classes=num_class)
    if args.cuda:
      model=model.cuda()
    if args.pretrained:
      model.load_state_dict(torch.load(args.pretrain_net))
      for param_tensor in model.state_dict():
       print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    lr=args.learning_rate
    optimizer = key2opt[args.optim](model.parameters(), lr=args.learning_rate)
    print('===> Training model')
    test_iou=-0.1
    for epoch in range(args.start_epoch,args.start_epoch+args.epochs+1):
      start=time.time()
      train_loss=train(epoch,args,model,training_data_loader,optimizer)
      train_end=time.time()
      train_time=train_end-start
      avg_test_loss,iou,acc_meter=test(args,model,testing_data_loader,optimizer)
      test_end=time.time()
      test_time=test_end-train_end
      print(train_time)
      print(test_time)
      if (iou[1] >test_iou):
        test_iou=iou[1]
        checkpoint(epoch,args,model)
      ResultPath=args.root_result+'/accuracy.txt'
      f = open(ResultPath, 'a+')
      new_content = '%d' % (epoch) + '\t' + '%.4f' % (train_loss)+ '\t' + '%.4f' % (avg_test_loss) + '\t' + '%.4f' % (acc_meter.average())  + '\t'  + '%.4f' % (iou[1])+ '\t' +'\n'
      f.write(new_content)
      f.close()

# Training settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--trainbatchsize', default=4,type=int,
                        help="input batch size per gpu for training")
    parser.add_argument('--validationbatchsize', default=1,type=int,
                        help="input batch size per gpu for validation")
    parser.add_argument('--epochs', default=400,type=int,
                        help="epochs to train for")
    parser.add_argument('--learning_rate', default=0.0001,type=float,
                        help="learning rate")
    parser.add_argument('--threads', default=1,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=256,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=6,type=int,
                        help="color dimension of the input image") 
    parser.add_argument('--pretrained', default=True,
                        help='whether to load saved trained model')
    parser.add_argument('--pretrain_net', default='/data/shi/qingyu/cd/ourtgrs/checkpoint-batchsize4-learning_rate0.0001-optimizersgd/best_model.pth',
                        help='path of saved pretrained model')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')                            
    parser.add_argument('--root_dataset', default='/data/shi/qingyu/cd/inria_data',
                        help='path of datasets')
    parser.add_argument('--optim', default='sgd',
                        help='optimizer')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--checkpoint', default='./checkpointinria00001',
                        help='folder to output checkpoints')
    parser.add_argument('--target_mode', default='cd',
                        help='folder (mode) of target label')
    parser.add_argument('--root_result', default='./resultinria00001',
                        help='path of result')
    args = parser.parse_args()
    args.checkpoint += '-batchsize' + str(args.trainbatchsize)
    args.checkpoint += '-learning_rate' + str(args.learning_rate)
    args.checkpoint += '-optimizer' + str(args.optim)
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.isdir(args.root_result):
        os.makedirs(args.root_result)
    main(args)