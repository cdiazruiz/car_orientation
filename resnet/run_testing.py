import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import finetuning_ResNet_regression as rn
from finetuning_ResNet_regression import ObsDataset
import json
import time
from scipy import stats
from itertools import compress
import pandas as pd
import matplotlib.gridspec as gridspec

def initializeModelandData(model_name, model_path, text_file_val, directory_val):
    # model_name = "resnet" #Use resnet for resnet18 or densenet for densenet121
    obs_model, input_size = rn.initialize_model(model_name=model_name,num_classes=1,\
    feature_extract=False, use_pretrained=False)
    obs_model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obs_model=obs_model.to(device)
    obs_model.eval()
    ##### Data loading #####

    #define transformations to use on images
    transformations=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    #test_set
    image_datasets={}
    batch_size=256
    image_datasets['Val']=ObsDataset(csv_path=text_file_val,
                                        root_dir=directory_val,
                                        transform=transformations)
    dataloaders_dict={}

    dataloaders_dict['Val'] = DataLoader(image_datasets['Val'], batch_size=batch_size ,shuffle=False, num_workers=4)

    #########
    return obs_model, dataloaders_dict
def eval_model(model, dataloaders_dict):
    was_training = model.training
    model.eval()
    images_so_far = 0
    errors=[]
    gt_labels=[]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for ibatch, sample_batched in enumerate(dataloaders_dict['Val']):
            inputs=sample_batched['image']
            labels=sample_batched['angle_obs']
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels=[p.cpu().numpy() for p in labels]
            outputs = model(inputs)
            preds_vec=[m.cpu().numpy() for m in outputs]
            preds_vec=np.asarray(preds_vec)
            preds_vec = ( preds_vec + np.pi) % (2 * np.pi ) - np.pi
            true_labels=np.asarray(true_labels)
            # print('true_labels', true_labels)
            # print('predictions', preds_vec)
            delta=np.abs(preds_vec.T-true_labels)
            # print('delta', delta)
            error=np.minimum(delta,np.abs(2*np.pi-delta))
            # print('error', error)
            # raise notImplemented
            errors.append(error)
            gt_labels.append(np.asarray(true_labels))
        return errors, gt_labels
def plot_histograms(errors,gt_labels,box_name, mean_name):
    # print('hola')
    #Histogram generation
    errors=np.hstack(np.hstack(errors))
    print('number of datapoints:', len(errors))
    gt_labels=np.concatenate(np.asarray(gt_labels))
    num_bins=19
    bins=np.arange(-np.pi,np.pi+2*np.pi/num_bins,2*np.pi/num_bins)
    n, bins, patches = plt.hist(gt_labels,bins ,edgecolor='k', weights=errors)
    index=np.digitize(gt_labels, bins)
    # print('index',index)
    # plt.xlabel('Angle of observation (rads)')
    # plt.ylabel('Sum of abs error')
    bin_means, bin_edges, binnumber = stats.binned_statistic(gt_labels,errors,statistic='mean', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    # print(bin_means)
    # plt.figure()
    # plt.hist(gt_labels, bins=bins,weights=errors, histtype='stepfilled',label='histogram of data')
    # plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,label='binned statistic of data')
    # # >plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
    # plt.legend(fontsize=10)
    # plt.ylim(0, 50)
    # plt.figure()
    # plt.scatter(gt_labels,errors, label='Data')
    # plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,label='Mean')
    # plt.xlabel('Angle of observation (rads)')
    # plt.ylabel('Error (absolute) rads')
    # plt.legend(fontsize=10)

    fig, ax = plt.subplots()
    list_errors=[]
    index=np.digitize(gt_labels, bins)
    for value in range(len(bins)):
        mask=np.equal(index,value)
        binned_error= list(compress(errors, mask))
        list_errors.append(binned_error)
    boxdict = ax.boxplot(list_errors)
    ax.set_title('Model error as a function of angle')
    ax.set_xlabel('Angle (rads)')
    ax.set_ylabel('Error abs (rads)')
    # plt.xticks(bin_means)
    labels=[bin_center for bin_center in bin_centers]
    labels= ['%.2f' % label for label in bin_centers]
    print(labels)

    # fliers = boxdict['fliers']
    # print('fliers', fliers)
    olier_count=0
    # print(boxdict['fliers'][0].get_ydata())
    for i in range(len(fliers)):
        # print('fliers',boxdict['fliers'][i].get_ydata())
        olier_count+= len(boxdict['fliers'][i].get_ydata())
    print('oo count', olier_count)

    # bin_means= ['%.2f' % val for val in bin_means]
    # plt.xticks(np.arange(min(bin_means), max(bin_means), np.pi/10))
    # .ylim(0,1)
    # ax.legend(['Grid Search'])
    # x = [0,5,9,10,15]
    # y = [0,1,2,3,4]
    # fig2, ax2= plt.subplots()
    # ax2.plot(x,y)
    # ax2.xticks(np.arange(min(x), max(x)+1, 1.0))
    plt.show()

def plot_barmean(errors1,errors2, gt_labels,box_name, mean_name):
    errors1=np.hstack(np.hstack(errors1))
    errors2=np.hstack(np.hstack(errors2))
    # print('number of datapoints:', len(errors))
    gt_labels=np.concatenate(np.asarray(gt_labels))
    num_bins=19
    bins=np.arange(-np.pi,np.pi+2*np.pi/num_bins,2*np.pi/num_bins)
    n, bins1, patches = plt.hist(gt_labels,bins ,edgecolor='k', weights=errors1)
    n, bins2, patches = plt.hist(gt_labels,bins ,edgecolor='k', weights=errors2)
    index=np.digitize(gt_labels, bins)
    #mean statistic
    bin_means1, bin_edges1, binnumber1 = stats.binned_statistic(gt_labels,errors1,statistic='mean', bins=bins)
    bin_means2, bin_edges2, binnumber2 = stats.binned_statistic(gt_labels,errors2,statistic='mean', bins=bins)
    bin_width = (bin_edges1[1] - bin_edges1[0])
    bin_centers = bin_edges1[1:] - bin_width/2
    labels=[bin_center for bin_center in bin_centers]

    labels= ['%.1f' % label for label in bin_centers]
    # labels= [' ' for i in range(len(labels)) if i%2==0]
    for i in range(len(labels)):
        if i%2==0:
            labels[i]=' '
    list_errors1=[]
    list_errors2=[]
    for value in range(len(bins)):
        mask=np.equal(index,value)
        binned_error1= list(compress(errors1, mask))
        binned_error2= list(compress(errors2, mask))
        list_errors1.append(binned_error1)
        list_errors2.append(binned_error2)

    #editing xticks
    label1=[]
    label1.append(' ')
    label1.append(' ')
    label1.extend(labels)
    label1[2]='%.0f' % np.round(bin_centers[0])
    label1[-1]='%.0f' % np.round(bin_centers[-1])

    #gnerating boxplots

    fig, ax1 = plt.subplots()
    boxdict1 = ax1.boxplot(list_errors1)
    # ax1.set_title('Resnet error as a function of angle')
    ax1.set_xlabel('Angle (rads)',fontsize=12)
    ax1.set_ylabel('Error abs (rads)',fontsize=12)
    plt.xticks(list(range(0,len(list_errors1)+2)), label1)
    fig, ax2 = plt.subplots()
    boxdict2 = ax2.boxplot(list_errors2)
    # ax2.set_title('Densenet error as a function of angle')
    ax2.set_xlabel('Angle (rads)',fontsize=12)
    ax2.set_ylabel('Error abs (rads)',fontsize=12)
    plt.xticks(list(range(0,len(list_errors2)+2)), label1)
    median1=[]
    median2=[]
    #counting outliers
    fliers1 = boxdict1['fliers']
    olier_count=0
    for i in range(len(fliers1)):
        olier_count+= len(boxdict1['fliers'][i].get_ydata())
    print('outlier count resnet', olier_count)
    fliers2 = boxdict2['fliers']
    olier_count=0
    for i in range(len(fliers2)):
        olier_count+= len(boxdict2['fliers'][i].get_ydata())
    print('outlier count densenet', olier_count)

    for medline in boxdict1['medians']:
        linedata = medline.get_ydata()
        median1.append(linedata[0])
    for medline in boxdict2['medians']:
        linedata = medline.get_ydata()
        median2.append(linedata[0])
    #median plot
    df = pd.DataFrame({'resnet': median1[1:],'densenet': median2[1:]}, index=labels)
    ax3 = df.plot.bar(rot=0)
    # ax3.set_title('Model error median as a function of angle')
    ax3.set_xlabel('Angle (rads)',fontsize=12)
    ax3.set_ylabel('Error abs (rads)',fontsize=12)
    plt.xticks(list(range(0,len(list_errors1))), label1[2:])
    #mean plot
    df1 = pd.DataFrame({'resnet':bin_means1,'densenet': bin_means2}, index=labels)
    ax4 = df1.plot.bar(rot=0)
    # ax4.set_title('Model error mean as a function of angle')
    ax4.set_xlabel('Angle (rads)',fontsize=12)
    ax4.set_ylabel('Error abs (rads)',fontsize=12)
    plt.xticks(list(range(0,len(list_errors1))), label1[2:])
    plt.show()

def visualize_model(model, dataloaders_dict, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    plot_idx=1
    fig = plt.figure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    switch=False
    count=0
    angles=[]
    preds=[]
    trues=[]
    id=0
    with torch.no_grad():
        for ibatch, sample_batched in enumerate(dataloaders_dict['Val']):
            inputs=sample_batched['image']
            labels=sample_batched['angle_obs']
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)
            inputs = inputs.to(device)
            labels = labels.to(device)
            true_labels=[p.cpu().numpy() for p in labels]
            outputs = model(inputs)
            preds_vec=[m.cpu().numpy() for m in outputs]
            # rand_idx=np.random.randint(num_images, size=num_images)
            # preds_vec=[preds_vec[index] for index in rand_idx]
            # true_labels=[true_labels[index] for index in rand_idx]
            # inputs=inputs[inputs[index] for index in rand_idx]
            letters='abcdefghijklmnopasdsafdsagsadgfsdfsa'
            gs1 = gridspec.GridSpec(4, 3)
            gs1.update(wspace=0.025, hspace=0.05)
            # ids=[0,1]
            # idr=4

            caca=[0,2,96,0,0,0,104,87,140]
            lista=[6,8,11,44,111,128,12]
            # caca=range()
            # caca=range(24*idr,(idr+1)*24)
            # for j in range(inputs.size()[0]):
            print('num images',inputs.size()[0])
            # for j in caca:
            while images_so_far<=num_images:
                j=lista[images_so_far]
                print('count',count)
                if not(switch):
                    print('subplot image')
                    ax = plt.subplot((num_images//3)*2, 3, plot_idx)
                    # print(ax)
                    # ax.axis('off')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plot_idx+=1
                    # ax.set_title('predicted: {}'.format(preds_vec[j]))
                    ax.set_title('('+letters[images_so_far]+')',fontsize=8)
                    plt.tight_layout()
    #                 plt.tick_params(axis='x',which='both',bottom=False,top=False,         # ticks along the top edge are off
    # labelbottom=False)
                    # print('Predicted:', preds_vec[j][0], 'GroundTruth:',true_labels[j] )
                    preds.append(preds_vec[j][0])
                    trues.append(true_labels[j])
                    plt.subplots_adjust(wspace=0)
                    id=0
                # print(np.shape(inputs.cpu().data[j].numpy()))
                    # angles.append()
                    img=inputs.cpu().data[j]
                    plt.imshow(img.permute(1, 2, 0).numpy())
                    images_so_far += 1
                    count+=1
                    if count%3==0:
                        switch=not(switch)
                elif switch:
                    print('subplot circle')
                    # print('plt idx', plot_idx)
                    ax = plt.subplot((num_images//3)*2, 3, plot_idx,frame_on=True)
                    gridspec_kw = {'width_ratios':[3, 1]}
                    circle1 = plt.Circle((0, 0.4), 0.5, color='k', fill=False)
                    # an=-preds_vec[count-3][0]
                    an=-preds[id]
                    an1=-trues[id]
                    ax.plot([0,0.5*np.cos(an)], [0.3,0.4+0.5*np.sin(an)],'k')
                    ax.plot([0,0.5*np.cos(an1)], [0.3,0.4+0.5*np.sin(an1)],'r--')
                    ax.set_xlim([-1, 1])
                    ax.set_ylim([-1, 1])
                    # ax.set_ybound([0,1])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.tight_layout()
                    id+=1



                    # ax.plot((0.5), (0.5), 'o', color='y')

                    ax.add_artist(circle1)
                    # ax.plot([0.5,0.5+0.1*np.cos(an)], [0.5,0.5+0.1*np.sin(an)])
                    # line=plt.plot([0.5,0.5+0.1*np.cos(an)], [0.5,0.5+0.1*np.sin(an)])
                    # ax.add_artist(line)

                    # ax.axis('equal')
                    # plt.axis('equal')
                    ax.axis('off')
                    plot_idx+=1
                    count+=1
                    if count%3==0:
                        switch=not(switch)
                        preds=[]
                        trues=[]

                if count==num_images*2:
                    left  = 0.1  # the left side of the subplots of the figure
                    right = 0.9    # the right side of the subplots of the figure
                    bottom = 0.1   # the bottom of the subplots of the figure
                    top = 0.9     # the top of the subplots of the figure
                    wspace = 0.01   # the amount of width reserved for space between subplots,
                                   # expressed as a fraction of the average axis width
                    hspace = 0.01   # the amount of height reserved for space between subplots,
                                   # expressed as a fraction of the average axis height
                    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                    # ax.figure(figsize=(1,2))
                    # ax.legend()
                    plt.legend(['Prediction','GroundTruth'],bbox_to_anchor=(1.25, 0.3), loc=1, borderaxespad=0. , prop={'size': 8})

                    fig=plt.gcf()
                    fig.set_figheight(4.0)
                    fig.set_figwidth(3.1)
                    plt.show()
                    # from matplotlib.pyplot import figure
                    # figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
                    return
                # images_so_far += 1
                # ax = plt.subplot(num_images//3, 3, images_so_far)
                # ax.axis('off')
                # print('caca')
                # ax.set_title('predicted: {}'.format(preds_vec[j]))
                # print('Predicted:', preds_vec[j][0], 'GroundTruth:',true_labels[j] )
                # print(np.shape(inputs.cpu().data[j].numpy()))
                # img=inputs.cpu().data[j]
                # plt.imshow(img.permute(1, 2, 0).numpy())
                # if images_so_far == num_images:
                    # model.train(mode=was_training)

        # model.train(mode=was_training)
def main():
    box_name='boxplot.png'
    mean_name='mean.png'
    #load trained weights  into angle regression model
    model_path_d='/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/samples/resnet/densenet/best_densenet.pt'
    model_path_r='/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/samples/resnet/best_resnet.pt'
    #define regression model
    text_file_val='/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/TestSplit_Obs/orientation_labels.txt'
    directory_val='/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/TestSplit_Obs/'
    obs_model_r, dataloaders_dict=initializeModelandData(model_name='resnet', model_path=model_path_r,\
     text_file_val=text_file_val,directory_val=directory_val)
    obs_model_d, dataloaders_dict=initializeModelandData(model_name='densenet', model_path=model_path_d,\
    text_file_val=text_file_val,directory_val=directory_val)
    obs_model_r.load_state_dict(torch.load(model_path_r))
    obs_model_d.load_state_dict(torch.load(model_path_d))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obs_model_r=obs_model_r.to(device)
    obs_model_r.eval()
    obs_model_d=obs_model_d.to(device)
    obs_model_d.eval()

    #uncomment to run boxplots and media bar plot
    errors1, gt_labels=eval_model(obs_model_r, dataloaders_dict)
    errors2, gt_labels=eval_model(obs_model_d, dataloaders_dict)
    plot_barmean(errors1,errors2, gt_labels,box_name=box_name, mean_name=mean_name)
    # # plot_histograms(errors,gt_labels,box_name,mean_name)

    #run visualization
    # visualize_model(obs_model_r, dataloaders_dict, num_images=6)

if __name__ == "__main__":
    main()
