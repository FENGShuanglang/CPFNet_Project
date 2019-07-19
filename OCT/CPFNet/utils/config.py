# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:52 2019

@author: Administrator
"""
class DefaultConfig(object):
    num_epochs=80
    epoch_start_i=0
    checkpoint_step=5
    validation_step=1
    crop_height=256
    crop_width=256
    batch_size=6   
    
    
    data='/home/FENGsl/CPFNet_Project/dataset'
    dataset="OCT/trainingset"
    log_dirs='/home/FENGsl/CPFNet_Project/Log/OCT'
    
    lr=0.01    
    lr_mode= 'poly'
    net_work= 'BaseNet'
    momentum = 0.9#
    weight_decay = 1e-4#


    mode='test'
    num_classes=4

    
    k_fold=4
    test_fold=4
    num_workers=4
    
    cuda='1'
    use_gpu=True
    pretrained_model_path='/home/FENGsl/CPFNet_Project/OCT/CPFNet/checkpoints/model_004_0.7985.pth.tar'
    save_model_path='./checkpoints'
    


