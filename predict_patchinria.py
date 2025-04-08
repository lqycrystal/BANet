import torch
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
#from tiramisu_de import FCDenseNet67
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from PIL import Image
import cv2
import h5py
from sklearn.metrics import confusion_matrix
from skimage import io
from unet_model import UNet
class generateDataset(Dataset):

        def __init__(self, dirFiles,img_size,colordim,isTrain=True):
                self.isTrain = isTrain
                self.dirFiles = dirFiles
                self.nameFiles = [name for name in os.listdir(dirFiles) if os.path.isfile(os.path.join(dirFiles, name))]
                self.numFiles = len(self.nameFiles)
                self.img_size = img_size
                self.colordim = colordim
                print('number of files : ' + str(self.numFiles))
                
        def __getitem__(self, index):
                filename = self.dirFiles + self.nameFiles[index]
                imga=io.imread(filename)
                img=np.zeros((256,256,6))
                img[:,:,0:3]=imga[:,0:256,:]
                img[:,:,3:6]=imga[:,256:512,:]
                img=img/255.0
                img = torch.from_numpy(img).float()
                img = img.transpose(0, 1).transpose(0, 2)
                imgName, imgSuf = os.path.splitext(self.nameFiles[index])
                return img, imgName
        
        def __len__(self):
                return int(self.numFiles)

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
    predDataset = generateDataset(args.pre_root_dir, args.img_size, args.colordim, isTrain=False)
    predLoader = DataLoader(dataset=predDataset, batch_size=args.predictbatchsize, num_workers=args.threads)
    with torch.no_grad():
      cm_w = np.zeros((2,2))
      for batch_idx, (batch_x, batch_name) in enumerate(predLoader):
        batch_x = batch_x
        if args.cuda:
            batch_x = batch_x.cuda()
        out= model(batch_x)
        pred_prop, pred_label = torch.max(out[5], 1)
        pred_prop_np = pred_prop.cpu().numpy()
        pred_label_np = pred_label.cpu().numpy() 
        #print(len(batch_name))       
        for id in range(len(batch_name)):
                pred_label_single = pred_label_np[id, :, :]
                predLabel_filename = args.preDir +  batch_name[id] + '.png'
                
                label_filename= args.label_root_dir +  batch_name[id] + '.png'
                label = io.imread(label_filename)
                label=np.where(label>0,1,0)
                cm = confusion_matrix(label.ravel(), pred_label_single.ravel())
                print(cm)
                cm_w = cm_w + cm
                pred_label_single=np.where(pred_label_single>0,255,0)
                cv2.imwrite(predLabel_filename, pred_label_single.astype(np.uint8))
                #OA_s, F1_s, IoU_s = evaluate(cm)
                #print('OA_s = ' + str(OA_s) + ', F1_s = ' + str(F1_s) + ', IoU = ' + str(IoU_s))
        
      print(cm_w)      
      OA_w, F1_w, IoU_w = evaluate(cm_w)
      print('OA_w = ' + str(OA_w) + ', F1_w = ' + str(F1_w) + ', IoU = ' + str(IoU_w))

def evaluate(cm):

        UAur=float(cm[1][1])/float(cm[1][0]+cm[1][1])
        UAnonur=float(cm[0][0])/float(cm[0][0]+cm[0][1])
        PAur=float(cm[1][1])/float(cm[0][1]+cm[1][1])
        PAnonur=float(cm[0][0])/float(cm[1][0]+cm[0][0])
        OA=float(cm[1][1]+cm[0][0])/float(cm[1][0]+cm[1][1]+cm[0][0]+cm[0][1])
        F1=2*UAur*PAur/(UAur+PAur)
        IoU=float(cm[1][1])/float(cm[1][0]+cm[1][1]+cm[0][1])
        
        return OA, F1, IoU


# Prediction settings
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0,type=int,
                        help="a name for identifying the model")
    parser.add_argument('--cuda', default=True,
                        help="a name for identifying the model")
    parser.add_argument('--predictbatchsize', default=1,type=int,
                        help="input batch size per gpu for prediction")
    parser.add_argument('--threads', default=1,type=int,
                        help="number of threads for data loader to use")
    parser.add_argument('--img_size', default=256,type=int,
                        help="image size of the input")
    parser.add_argument('--seed', default=123,type=int,
                        help="random seed to use")
    parser.add_argument('--colordim', default=6,type=int,
                        help="color dimension of the input image") 
    parser.add_argument('--pretrain_net', default='./checkpointinria00001-batchsize4-learning_rate0.0001-optimizersgd/best_model.pth',
                        help='path of saved pretrained model')                       
    parser.add_argument('--pre_root_dir', default='/data/shi/qingyu/cd/inria_data/test/osmimage/',
                        help='path of input datasets for predict')
    parser.add_argument('--label_root_dir', default='/data/shi/qingyu/cd/inria_data/test/cd/',
                        help='path of label of input datasets')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--preDir', default='./predictionourinria00001/',
                        help='path of result')
    args = parser.parse_args()

    if not os.path.isdir(args.preDir):
        os.makedirs(args.preDir)
    main(args)