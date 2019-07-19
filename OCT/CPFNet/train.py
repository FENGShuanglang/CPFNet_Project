import argparse
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.OCT import OCT
import socket
from datetime import datetime
import os
from model.BaseNet import CPFNet
from model.unet import UNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from  PIL import Image

import utils.utils as u
import utils.loss as LS
from utils.config import DefaultConfig
import torch.backends.cudnn as cudnn
def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')
       

        total_Dice=[]
        total_Dice1=[]
        total_Dice2=[]
        total_Dice3=[]
        total_Dice.append(total_Dice1)
        total_Dice.append(total_Dice2)
        total_Dice.append(total_Dice3)
        Acc=[]
        
        cur_cube=[]
        cur_label_cube=[]
        next_cube=[]
        counter=0
        end_flag=False

        for i, (data, labels) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = labels[0].cuda()
            slice_num=labels[1][0].long().item()
            
            # get RGB predict image
            
            aux_predict,predicts = model(data)
            predict=torch.argmax(torch.exp(predicts),dim=1)
            batch_size=predict.size()[0]

            counter+=batch_size
            if counter<=slice_num:
                cur_cube.append(predict)
                cur_label_cube.append(label)
                if counter==slice_num:
                    end_flag=True
                    counter=0
            else:
                last=batch_size-(counter-slice_num)

                last_p=predict[0:last]
                last_l=label[0:last]

                first_p=predict[last:]
                first_l=label[last:]

                cur_cube.append(last_p)
                cur_label_cube.append(last_l)
                end_flag=True
                counter=counter-slice_num

            if end_flag:
                end_flag=False
                predict_cube=torch.stack(cur_cube,dim=0).squeeze()
                label_cube=torch.stack(cur_label_cube,dim=0).squeeze()
                cur_cube=[]
                cur_label_cube=[]
                if counter!=0:
                    cur_cube.append(first_p)
                    cur_label_cube.append(first_l)

                assert predict_cube.size()[0]==slice_num
                Dice,true_label,acc=u.eval_multi_seg(predict_cube,label_cube,args.num_classes)

                for class_id in range(args.num_classes-1):
                    if true_label[class_id]!=0:
                        total_Dice[class_id].append(Dice[class_id])
                Acc.append(acc)
                len0=len(total_Dice[0]) if len(total_Dice[0])!=0 else 1
                len1=len(total_Dice[1]) if len(total_Dice[1])!=0 else 1
                len2=len(total_Dice[2]) if len(total_Dice[2])!=0 else 1

                dice1=sum(total_Dice[0])/len0
                dice2=sum(total_Dice[1])/len1
                dice3=sum(total_Dice[2])/len2
                ACC=sum(Acc)/len(Acc)
                mean_dice=(dice1+dice2+dice3)/3.0
                tbar.set_description('Mean_D: %3f, Dice1: %.3f, Dice2: %.3f, Dice3: %.3f, ACC: %.3f' % (mean_dice,dice1,dice2,dice3,ACC))
        print('Mean_Dice:',mean_dice)
        print('Dice1:',dice1)
        print('Dice2:',dice2)
        print('Dice3:',dice3)
        print('Acc:',ACC)
      
        return mean_dice,dice1,dice2,dice3,ACC

def train(args, model, optimizer,criterion, dataloader_train, dataloader_val):
    comments=os.getcwd().split('/')[-1]
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.log_dirs, comments+'_'+current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    step = 0
    best_pred=0.0
    for epoch in range(args.num_epochs):
        lr = u.adjust_learning_rate(args,optimizer,epoch) 
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        train_loss=0.0
#        is_best=False
        for i,(data, label) in enumerate(dataloader_train):
            # if i>9:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda().long()
            optimizer.zero_grad()
            aux_out,main_out = model(data)
            # get weight_map
            weight_map=torch.zeros(args.num_classes)
            weight_map=weight_map.cuda()
            for ind in range(args.num_classes):
                weight_map[ind]=1/(torch.sum((label==ind).float())+1.0)
            # print(weight_map)

            loss_aux=F.nll_loss(main_out,label,weight=None)
            loss_main= criterion[1](main_out, label)

            loss =loss_main+loss_aux
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss/(i+1)))
            step += 1
            if step%10==0:
                writer.add_scalar('Train/loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        

        if epoch % args.validation_step == 0:
            mean_Dice,Dice1,Dice2,Dice3,acc= val(args, model, dataloader_val)
          
            writer.add_scalar('Valid/Mean_val', mean_Dice, epoch)
            writer.add_scalar('Valid/Dice1_val', Dice1, epoch)
            writer.add_scalar('Valid/Dice2_val', Dice2, epoch)
            writer.add_scalar('Valid/Dice3_val', Dice3, epoch)
            writer.add_scalar('Valid/Acc_val', acc, epoch)

            # mean_Dice=(Dice1+Dice2+Dice3)/3.0
            is_best=mean_Dice > best_pred
            best_pred = max(best_pred, mean_Dice)
            checkpoint_dir = args.save_model_path
            # checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest =os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_pred,
                    }, best_pred,epoch,is_best, checkpoint_dir,filename=checkpoint_latest)
                    
def test(model,dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(dataloader,desc='\r')
        tq.set_description('test')
        comments=os.getcwd().split('/')[-1]
        for i, (data, label_path) in enumerate(tq):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            aux_pred,predict = model(data)
            predict=torch.argmax(torch.exp(predict),dim=1)
            pred=predict.data.cpu().numpy()
            pred_RGB=OCT.COLOR_DICT[pred.astype(np.uint8)]
            
            for index,item in enumerate(label_path):
                save_img_path=label_path[index].replace('test_mask',comments+'_mask')
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                img=Image.fromarray(pred_RGB[index].squeeze().astype(np.uint8))
                img.save(save_img_path)
                tq.set_postfix(str=str(save_img_path))
        tq.close()
            
def main(mode='train',args=None):


    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    dataset_val = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )


    dataset_test = OCT(dataset_path, scale=(args.crop_height, args.crop_width),mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        # this has to be 1
        batch_size=len(args.cuda.split(',')),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False 
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    
    
    #load model
    model_all={'BaseNet':CPFNet(out_planes=args.num_classes),'UNet':UNet(in_channels=1, n_classes=args.num_classes)}
    model=model_all[args.net_work]
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_model_path and mode=='test':
        print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
        
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_aux=nn.NLLLoss(weight=None)
    criterion_main=LS.Multi_DiceLoss(class_num=args.num_classes)
    criterion=[criterion_aux,criterion_main]
    if mode=='train':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val)
    if mode=='test':
        test(model,dataloader_test, args)
    if mode=='train_test':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val)
        test(model,dataloader_test, args)

if __name__ == '__main__':
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()

    modes=args.mode

    if modes=='train':
        main(mode='train',args=args)
    elif modes=='test':
        main(mode='test',args=args)
    elif modes=='train_test':
        main(mode='train_test',args=args)


