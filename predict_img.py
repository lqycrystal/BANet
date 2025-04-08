import numpy as np
import torch

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch.nn as nn
import argparse
from unet_model import UNet
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image
import cv2
import h5py
from sklearn.metrics import confusion_matrix
from skimage import io
import itertools


  
def main(args):
    if args.cuda and not torch.cuda.is_available():
      raise Exception("No GPU found, please run without --cuda")
    num_class=args.num_class
    if args.id==0:
       model=UNet(n_channels=args.colordim,n_classes=num_class)
    if args.cuda:
      model=model.cuda()
    model.load_state_dict(torch.load(args.pretrain_net))
    model.eval()


    batch_size = args.predictbatchsize

    #print(len(list(val_images)))
    #print(len(list(val_labels)))
    
    #cm_w = np.zeros((2,2))
    
    with torch.no_grad():
      print('Inferencing begin')
      src = io.imread(args.pred_rgb_file).astype('float32') 
      src2 = io.imread(args.pred_osm_file).astype('float32') 
      src2=np.where(src2>0,255,0)
      src2new=np.zeros((src2.shape[0],src2.shape[1],3))
      src2new[:,:,0]=np.copy(src2)
      src2new[:,:,1]=np.copy(src2)
      src2new[:,:,2]=np.copy(src2)
      #img,_=ah(src,1.5)
      #ndom = (np.load(args.pred_ndom_file).astype('float32')+16.86) / (39.81+16.86)
      #ndom_new=ndom[ ..., np.newaxis]
      img=np.concatenate((src, src2new), axis=2)
      img=img/255.0
      pred = np.zeros(img.shape[:2] + (num_class,))
      batch_total = count_sliding_window(img, step=args.step, window_size=(args.img_size,args.img_size)) // batch_size
      print('Total Batch : ' + str(batch_total))

      for batch_idx, coords in enumerate(grouper(batch_size, sliding_window(img, step=args.step, window_size=(args.img_size,args.img_size)))):

                image_patches = [np.copy(img[x:x+w, y:y+h]).transpose((2,0,1)) for x,y,w,h in coords]
                image_patches = np.asarray(image_patches)
                image_patches = torch.from_numpy(image_patches).cuda()

                outs = model(image_patches.float())
                outs_np = outs[5].detach().cpu().numpy()

                for out, (x, y, w, h) in zip(outs_np, coords):
                    out = out.transpose((1,2,0))
                    pred[x:x+w, y:y+h] += out

                        #progress_bar(batch_idx, batch_total)
      pred = np.argmax(pred, axis=-1)
      pred=np.where(pred>0,255,0)
      cv2.imwrite(args.pre_result, pred.astype(np.uint8))
            
def evaluate(cm):

        UAur=float(cm[1][1])/float(cm[1][0]+cm[1][1])
        UAnonur=float(cm[0][0])/float(cm[0][0]+cm[0][1])
        PAur=float(cm[1][1])/float(cm[0][1]+cm[1][1])
        PAnonur=float(cm[0][0])/float(cm[1][0]+cm[0][0])
        OA=float(cm[1][1]+cm[0][0])/float(cm[1][0]+cm[1][1]+cm[0][0]+cm[1][0])
        F1=2*UAur*PAur/(UAur+PAur)
        IoU=float(cm[1][1])/float(cm[1][0]+cm[1][1]+cm[0][1])
        
        return OA, F1, IoU

def sliding_window(img, step=128, window_size=(256,256)):

    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def grouper(n, iterable):

    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def count_sliding_window(img, step=128, window_size=(256, 256)):

    """ Count the number of windows in an image """
    nSW = 0
    for x in range(0, img.shape[0], step):
        if x + window_size[0] > img.shape[0]:
            x = img.shape[0] - window_size[0]
        for y in range(0, img.shape[1], step):
            if y + window_size[1] > img.shape[1]:
                y = img.shape[1] - window_size[1]
            nSW += 1

    return nSW

# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int,
                        help="a name for identifying the model")
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1, type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1, type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--colordim', default=6, type=int,
                        help="the channels of patch")
    parser.add_argument('--img_size', default=256, type=int,
                        help="patch size of the input")
    parser.add_argument('--step', default=124, type=int,
                        help="the overlap between neigbouring patch")
    parser.add_argument('--seed', default=123, type=int,
                        help="random seed to use")
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--pretrain_net', default='./checkpointinria001-batchsize4-learning_rate0.0001-optimizersgd/best_model.pth',help='path of saved pretrained model')                       
    parser.add_argument('--pred_rgb_file', default='/work/shi/DeepLearning/datasets/original/public/INRIA/AerialImageDataset/raw/train/images/austin13.tif',
                        help='the name of input rgb datasets for predict')
    parser.add_argument('--pred_osm_file', default='/data/shi/qingyu/cd/austin/osm/austin13_osma.tif',
                        help='the name of input rgb datasets for predict')
    parser.add_argument('--pre_result', default='austin13ournew.png',
                        help='the name of predicted result')
    args = parser.parse_args()

    main(args)