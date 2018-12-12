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
import math
from PIL import Image
warnings.filterwarnings('ignore')
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def show_pic(image, angle):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.title(str(angle))
#     plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated

class ObsDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, sep=" ", header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 2])
        # Third column is for an operation indicator
        self.orientation_label = np.asarray(self.data_info.iloc[:, 3])
        # Calculate len
        self.data_len = len(self.data_info.index)
        self.root_dir=root_dir

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = os.path.join(self.root_dir,\
        self.image_arr[index])
        # Open image
        img_as_img = Image.open(single_image_name)
        # If there is an transformation
        if transform is not None:
            img_as_tensor = self.transform(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        angle_obs = self.label_arr[index]
        angle_orient = self.orientation_label[index]
        sample = {'image': img_as_tensor, 'angle_obs': angle_obs, 'orientation':angle_orient}
        return sample

    def __len__(self):
        return self.data_len

class AnglesDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.angle_frame = pd.read_csv(csv_file,sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.angle_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.angle_frame.iloc[idx, 0])
        image = io.imread(img_name)
        angle = self.angle_frame.iloc[idx, 2]
        sample = {'image': image, 'angle': angle}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, angle = sample['image'], sample['angle']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

#         # h and w are swapped for landmarks because for images,
#         # x and y axes are axis 1 and 0 respectively
#         landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'angle': angle}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, angle = sample['image'], sample['angle']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if h-new_h<0:
            top=0
            left=0
        else:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]



        return {'image': image, 'angle': angle}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, angle = sample['image'], sample['angle']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'angle': torch.from_numpy(np.asarray(angle))}


# # Top level data directory. Here we assume the format of the directory conforms
# #   to the ImageFolder structure
# data_dir = "./orientation_data"
# # data_dir = "./hymenoptera_data"
# # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet"
#
# # Number of classes in the dataset
# num_classes = 1
#
# # Batch size for training (change depending on how much memory you have)
# batch_size = 16
#
# # Number of epochs to train for
# num_epochs = 200
#
# # Flag for feature extracting. When False, we finetune the whole model,
# #   when True we only update the reshaped layer params
# feature_extract = False

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_model(model, dataloaders, criterion, optimizer, fold_path, num_epochs=25, is_inception=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    val_acc_history = []
    train_acc=[]
    train_loss=[]
    val_loss=[]
    best_model_wts = copy.deepcopy(model.state_dict())
    model_last=copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss=300.00

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['Val','Train']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for ibatch, sample_batched in enumerate(dataloaders[phase]):
                inputs=sample_batched['image']
                labels=sample_batched['angle_obs']
                # print('input tensors shape:', inputs.size())
                # print('label tensors shape:', labels.size())
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)
                labels=labels.view(labels.size()[0], -1)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'Train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                #need to edit the running corrects
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc=0

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            if phase == 'Train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                with open(fold_path+ '/'+ 'train_loss' + ".txt", "a") as myfile:
                    myfile.write(str(epoch_loss)+"\n")
            # deep copy the model
            if phase == 'Val' and  epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'Val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
                with open(fold_path + '/' + 'val_loss' + ".txt", "a") as myfile:
                    myfile.write(str(epoch_loss)+"\n")
            if math.isnan(epoch_loss):
                break
        if math.isnan(epoch_loss):
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model_last_wts=copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_loss, val_loss, train_acc, model_last_wts, best_loss



# Initialize and Reshape the Networks
# -----------------------------------
# Resnet
# ~~~~~~
#
# Resnet was introduced in the paper `Deep Residual Learning for Image
# Recognition <https://arxiv.org/abs/1512.03385>`__. There are several
# variants of different sizes, including Resnet18, Resnet34, Resnet50,
# Resnet101, and Resnet152, all of which are available from torchvision
# models. Here we use Resnet18, as our dataset is small and only has two
# classes. When we print the model, we see that the last layer is a fully
# connected layer as shown below:
#
# ::
#
#    (fc): Linear(in_features=512, out_features=1000, bias=True)
#
# Thus, we must reinitialize ``model.fc`` to be a Linear layer with 512
# input features and 2 output features with:
#
# ::
#
#    model.fc = nn.Linear(512, num_classes)
#
# Densenet
# ~~~~~~~~
#
# Densenet was introduced in the paper `Densely Connected Convolutional
# Networks <https://arxiv.org/abs/1608.06993>`__. Torchvision has four
# variants of Densenet but here we only use Densenet-121. The output layer
# is a linear layer with 1024 input features:
#
# ::
#
#    (classifier): Linear(in_features=1024, out_features=1000, bias=True)
#
# To reshape the network, we reinitialize the classifier’s linear layer as
#
# ::
#
#    model.classifier = nn.Linear(1024, num_classes)
#


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Plot the training curves of validation accuracy vs. number
#  of training epochs for the transfer learning method and
#  the model trained from scratch
def plot_loss(train_hist, val_hist, filename):
    fig1=plt.figure()
    plt.title("Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    num_epochs=len(train_hist)
    # ohist = [h.cpu().numpy() for h in hist]
    plt.plot(range(1,num_epochs+1),train_hist,label="Training")
    plt.plot(range(1,num_epochs+1),val_hist,label="Validation")
    plt.ylim((0,3.5))
    plt.xticks(np.arange(1, num_epochs+1,25))
    plt.legend()
    # plt.show()
    fig1.savefig(filename, bbox_inches='tight')

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

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
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds_vec[j]))
                print('Predicted:', preds_vec[j][0], 'GroundTruth:',true_labels[j] )
                print(np.shape(inputs.cpu().data[j].numpy()))
                img=inputs.cpu().data[j]
                plt.imshow(img.permute(1, 2, 0).numpy())
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def eval_model(model, num_images=10):
    was_training = model.training
    model.eval()
    images_so_far = 0
    errors=[]
    gt_labels=[]
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
            true_labels=np.asarray(true_labels)
            error=np.abs(preds_vec.T-true_labels)
            errors.append(error)
            gt_labels.append(np.asarray(true_labels))
#             for j in range(inputs.size()[0]):
#                 images_so_far += 1
#                 if images_so_far == num_images:
#                     model.train(mode=was_training)
#                     return errors, gt_labels
        return errors, gt_labels
        model.train(mode=was_training)

def plot_histograms(model_ft):
    errors, gt_labels=eval_model(model_ft, num_images=33)
    errors=np.hstack(np.hstack(errors))
    gt_labels=np.concatenate(np.asarray(gt_labels))
    bin_means, bin_edges, binnumber = stats.binned_statistic(gt_labels,errors,statistic='mean', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    print(bin_means)
    plt.figure()
    plt.hist(gt_labels, bins=bins,weights=errors, histtype='stepfilled',label='histogram of data')
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,label='binned statistic of data')
    # >plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
    plt.legend(fontsize=10)
    plt.ylim(0, 50)
    plt.show()
    plt.figure()
    plt.scatter(gt_labels,errors, label='Data')
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=5,label='Mean')
    plt.xlabel('Angle of observation (rads)')
    plt.ylabel('Error (absolute) rads')
    plt.legend(fontsize=10)
