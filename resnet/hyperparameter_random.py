from __future__ import print_function
from __future__ import division
import pandas as pd
from skimage import io, transform
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import warnings
import finetuning_ResNet_regression as rn
from finetuning_ResNet_regression import ObsDataset
from PIL import Image
import datetime
import json

#### Initializing model
def run_training_trial(model_name, lr, optimizer, batch_size, reg,  fold_path,num_epochs = 75):
    # model_name = "resnet" #Use resnet for resnet18 or densenet for densenet121
    # optimizer = "sgd"  #sgd or adam
    #

    # # Batch size for training (change depending on how much memory you have)
    # batch_size = 32
    # Number of epochs to train for
    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    num_classes = 1  #just one class for  regression at the  final fc layer
    feature_extract = False
    print('Current folder path to save files:',fold_path)
    model_ft, input_size = rn.initialize_model(model_name, num_classes,\
     feature_extract, use_pretrained=True) #pretrained on imagenet
    # print(model_ft)
    #####    #########

    ##### Data loading
    #define transformations to use on images
    transformations=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
        # transforms.Normalize([69.285, 73.039, 78.033],[71.338, 71.758, 73.490])])

    #training data set
    text_file_train='orientation_data/regression/Train/orientation_labels.txt'
    directory_train='orientation_data/regression/Train/'
    image_datasets={'Train':ObsDataset(csv_path=text_file_train,
                                        root_dir=directory_train,
                                        transform=transformations)}
    #validation data set
    text_file_val='orientation_data/regression/Val/orientation_labels.txt'
    directory_val='orientation_data/regression/Val/'
    image_datasets['Val']=ObsDataset(csv_path=text_file_val,
                                        root_dir=directory_val,
                                        transform=transformations)
    dataloaders_dict = {'Train':DataLoader(image_datasets['Train'], batch_size=batch_size,shuffle=True, num_workers=4)}

    dataloaders_dict['Val'] = DataLoader(image_datasets['Val'], batch_size=batch_size ,shuffle=True, num_workers=4)
    # print(dataloaders_dict)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #########

    #########    Define optimization
    #Send the model to GPU
    model_ft = model_ft.to(device)
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.

    #This section can be commendted out $$$$$$$$$$$$$$$$$$$$$
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                # print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                continue
                # print("\t",name)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    # Observe that all parameters are being optimized
    if optimizer=="sgd":
        optimizer_ft = optim.SGD(params_to_update, lr=lr, weight_decay=reg, momentum=0.9)
    elif optimizer=="adam":
        optimizer_ft = optim.Adam(params_to_update, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=reg, amsgrad=False)
    criterion = nn.MSELoss()
    # Train and evaluate
    # now = datetime.datetime.now()
    # date_str=now.strftime("%Y-%m-%d_%H:%M")
    # name_model=model_name + '_' + optimizer + '_'+ date_str
    model_ft, hist, train_hist, val_hist, train_acc, model_last, model_loss_min =\
     rn.train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,fold_path,\
    num_epochs, is_inception=(model_name=="inception") )
    rn.plot_loss(train_hist, val_hist, fold_path + '/' + model_name +'.png')
    torch.save(model_ft.state_dict(), fold_path + '/' + model_name + '.pt')
    return model_loss_min #minimum loss for the validation epoch

def sample_lr(low=0.0001, high=0.01, size=None):
    a=np.log10(low)
    b=np.log10(high)
    return 10**(np.random.uniform(a, b, size))

def sample_weightdecay(low=0.0, high=0.001, size=None):
    flip1=np.random.randint(0,2,size) # first coin flip to determine if weight
    #decay is zero
    weight_decays=np.zeros(size)
    for count, x in enumerate(flip1):
        if x==0:
            weight_decays[count]=0
        else:
            weight_decays[count]=np.random.uniform(low,high,1)
    return weight_decays

def sample_opt(optimizer_names=['sgd','adam'],size=None):
    optimizers=np.random.choice(optimizer_names, size)
    return optimizers

def sample_batch_size(low=32, high=128, size=None):
    batch_sizes=np.random.randint(low,high+1,size)
    return batch_sizes

def main():
    model_name="densenet"
    trials_exp= 2 #number of trials per experiment
    expList=[5]   #experiment list
    num_folders=trials_exp
    root_dir='Results_Random_Hyp'
    foldExpList=[]
    foldTrialList=[]
    num_epochs=75

    for i, val in enumerate(expList):
        #creating folders for each experiment
        foldExpList.append(root_dir + '/' + 'Exp' + str(val))
        if not os.path.exists(foldExpList[i]):
            os.mkdir(foldExpList[i])

        #sampling for hyperparameters for each experiment
        lrList=sample_lr(0.0001,0.01,trials_exp)  #learning rate
        batch_sizeList=sample_batch_size(16,16,trials_exp)  #batchsize
        optList=sample_opt(['sgd','adam'],trials_exp) #optimizer scheme
        regList=sample_weightdecay(0,0.001,trials_exp) #weight decay parameter, acts as l2 regularization
        best_loss=np.inf
        param={} #this dic will hold best model for each experiment
        foldTrialList=[]
        for j in range(trials_exp):

            #creating folder for each trial
            foldTrialList.append(foldExpList[i] + '/' + str(j+1))
            if not os.path.exists(foldTrialList[j]):
                os.mkdir(foldTrialList[j])
            #hyper parameter values
            lr=lrList[j]
            opt=optList[j]
            batch_size=int(batch_sizeList[j])
            reg=regList[j]
            fold_path=foldTrialList[j]
            since = time.time() #start timer
            print('Parameters : lr: ', lr, 'opt: ', opt, 'bacth_size: ', batch_size, 'reg: ', reg)
            #running a training trial
            loss=run_training_trial(model_name=model_name, lr=lr, \
            optimizer=opt, batch_size=batch_size, reg=reg,\
            fold_path=fold_path,num_epochs=num_epochs)
            time_elapsed = time.time() - since

            #writing text file with parameters for trial
            with open(foldTrialList[j]+ '/'+ 'parameters' + ".txt", "a") as myfile:
                myfile.write('model_name ' + model_name +"\n"\
                +'lr ' + str(lr) +"\n"\
                +'batch_size ' + str(batch_size) +"\n"\
                +'opt ' +opt +"\n"\
                +'reg ' + str(reg) +"\n"\
                +'epochs '+ str(num_epochs) + "\n"\
                +'training time(m) '+ str(time_elapsed/60)) #in minutes

            #keeps track of best model
            if loss<best_loss:
                best_loss=loss
                param['model_name']=model_name
                param['lr']=lr
                param['batch_size']=batch_size
                param['opt']=opt
                param['reg']=reg
                param['epochs']=num_epochs
                param['loss']=loss
                param['training time']=time_elapsed/60
                param['folder_num']=j+1
            print('Trial ' + str(j+1) + ' Complete')
            print('Best Model So Far:', param)
        #writing info of the best model for each experiment
        with open(foldExpList[i]+'/'+'bestModelparams.txt','w' ) as file:
            file.write(json.dumps(param))

if __name__ == "__main__":
    main()
