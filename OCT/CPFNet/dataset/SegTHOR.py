import torch
import glob
import os
import sys

from torchvision import transforms
from torchvision.transforms import functional as F
#import cv2
from PIL import Image
# import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
#from utils import get_label_info, one_hot_it
import random
import skimage.io as io
from utils.config import DefaultConfig

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class SegTHOR(torch.utils.data.Dataset):
    Esophagus = [255,0,0]
    Heart = [0,255,0]
    Trachea = [0,0,255]
    Aorta = [255,255,0]
    Unlabelled=[0,0,0]
    COLOR_DICT = np.array([Unlabelled,Esophagus,Heart,Trachea, Aorta])
    def __init__(self, dataset_path,scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path=dataset_path+'/img'
        self.mask_path=dataset_path+'/mask'
        self.image_lists,self.label_lists=self.read_list(self.img_path)
        self.flip =iaa.SomeOf((1,4),[
             iaa.Fliplr(0.5),
             iaa.Flipud(0.5),
             iaa.Affine(rotate=(-15, 15)),
             iaa.ContrastNormalization((0.5, 1.5))], random_order=True)
        # resize
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()

        
        

        
       

    def __getitem__(self, index):
        # load image and crop
        img_cur = Image.open(self.image_lists[index])
        length=len(self.image_lists)
        if index==0:
            pre_index=0
        else:
            pre_index=index-1

        if index>=length-1:
            next_index=index
        else:
            next_index=index+1
        
        if self.image_lists[pre_index].split('/')[-2]==self.image_lists[index].split('/')[-2]:
            img_pre=Image.open(self.image_lists[pre_index])
        else:
            img_pre=img_cur

        if self.image_lists[next_index].split('/')[-2]==self.image_lists[index].split('/')[-2]:
            img_next=Image.open(self.image_lists[next_index])
        else:
            img_next=img_cur


        
        img = np.stack((img_pre,img_cur,img_next),axis=2).astype(np.uint16)  #2.5D
        labels=self.label_lists[index]
        #load label
        if self.mode !='test':
            label_ori = Image.open(self.label_lists[index])  
            label_ori = np.array(label_ori)
            label=np.ones(shape=(label_ori.shape[0],label_ori.shape[1]),dtype=np.uint8)

            #convert RGB  to one hot
            
            for i in range(len(self.COLOR_DICT)):
                equality = np.equal(label_ori, self.COLOR_DICT[i])
                class_map = np.all(equality, axis=-1)
                label[class_map]=i

            # augment image and label
            if self.mode == 'train' or self.mode == 'train_val' :
                seq_det = self.flip.to_deterministic()#固定变换
                segmap = ia.SegmentationMapOnImage(label, shape=label.shape, nb_classes=5)
                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)

            label_img=torch.from_numpy(label.copy()).float()
            if self.mode == 'val':
                img_num=len(os.listdir(os.path.dirname(labels)))
                labels=label_img,img_num
            else:
                labels=label_img
        imgs=img.transpose(2,0,1)/65535.0
        img = torch.from_numpy(imgs.copy()).float()#self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)
    def read_list(self,image_path):
        fold=sorted(os.listdir(image_path),key=lambda x: int(x[-2:]))
        # print(fold)
        os.listdir()
        img_list=[]
        if self.mode=='train':
            fold_r=fold[:32]
            # fold_r.remove('f'+str(k_fold_test))# remove testdata
            for item in fold_r:
                img_list+=sorted(glob.glob(os.path.join(image_path,item)+'/*.png'),key=lambda x: (int(x.split('/')[-2][-2:]),int(x.split('/')[-1].split('.')[0])))#two keys sorted
            # print(len(img_list))
            label_list=[x.replace('img','mask') for x in img_list]

        elif self.mode=='val':
            fold_s=fold[32:]
            for item in fold_s:

                img_list+=sorted(glob.glob(os.path.join(image_path,item)+'/*.png'),key=lambda x: (int(x.split('/')[-2][-2:]),int(x.split('/')[-1].split('.')[0])))
            label_list=[x.replace('img','mask') for x in img_list]

        elif self.mode=='train_val':
            fold_tv=fold
            for item in fold_tv:
                img_list+=sorted(glob.glob(os.path.join(image_path,item)+'/*.png'),key=lambda x: (int(x.split('/')[-2][-2:]),int(x.split('/')[-1].split('.')[0])))
            label_list=[x.replace('img','mask') for x in img_list]

        elif self.mode=='test':
            test_list=image_path.replace('img','test')
            fold_t=sorted(os.listdir(test_list),key=lambda x: int(x[-2:]))
            for item in fold_t:
                img_list+=sorted(glob.glob(os.path.join(test_list,item)+'/*.png'),key=lambda x: (int(x.split('/')[-2][-2:]),int(x.split('/')[-1].split('.')[0])))
            label_list=[x.replace('test','test_mask') for x in img_list]
                
        return img_list,label_list



                





# if __name__ == '__main__':
#    data = SegTHOR(r'G:\KeTi\JBHI_pytorch\Dataset\seg_test', (512, 512),mode='val')
#    # from model.build_BiSeNet import BiSeNet
#    # from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
#    from torch.utils.data import DataLoader
#    dataloader_test = DataLoader(
#        data,
#        # this has to be 1
#        batch_size=2,
#        shuffle=False,
#        num_workers=0,
#        pin_memory=True,
#        drop_last=True 
#    )
#    for i, (img, label) in enumerate(dataloader_test):

#        print(img.size())
      

#        print(label[0].size(),label[1][0])
#     #    g=img.data.numpy()
#     #    for index,item in enumerate(g):
#     #        im=item*65535
#     #        for inf,it in enumerate(im):
#     #             io.imsave(os.path.join(r'G:\KeTi\JBHI_pytorch\Dataset\ffffffffffffff',str(index)+'_'+str(inf)+'.png'),it.astype(np.uint16))


#     #    la=label.data.numpy()
#     #    for index,item in enumerate(la):
#     #        im=data.COLOR_DICT[item.astype(np.uint8)]
#     #        io.imsave(os.path.join(r'G:\KeTi\JBHI_pytorch\Dataset\ffffffffffffff',str(index)+'_'+str(inf)+'_seg'+'.png'),im)

#        if i>3:
#            break

   

