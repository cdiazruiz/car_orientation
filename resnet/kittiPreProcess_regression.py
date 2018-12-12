"""
    Generaes folder of KITTI dataset labelled by their observation angle
"""

# just importing stuff
import os
import sys
import cv2
import csv
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from skimage import io
import numpy as np



# Main code here:

# Just 10 data files for reading
# labelPath = '/media/vikram/DATA/ASL/VehicleOrientation/Code/labels'
# imgPath = '/media/vikram/DATA/ASL/VehicleOrientation/Code/images'

cropDir = '/home/cad297/Documents/ResNet18/classification_models-master/orientation_data/regression'
cropDir = '/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/TestObs/'
# Entire data files
labelPath = '/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/TestSplit2/label_2' #text files
imgPath = '/home/yh675/Desktop/Car_Orientation/Mask_RCNN-master/TestSplit2/image_2' #png files

# Labels for images
# labels = ['Train', 'Val'] # positive => [0, pi)  and  negative => [-pi, 0)

labels=['All']
filePath = {}

# Making the directory for storing the cropped files
if not os.path.exists(cropDir):
    os.makedirs(cropDir)

# Open all the files for writing
for labName in labels:
    fileName = cropDir + '/' + labName
    if not os.path.exists(fileName):
        os.makedirs(fileName)
        filePath[labName]= fileName

label=labels[0]
i=1
# print(filePath)
fh = open(filePath[label] + '/orientation_labels.txt','w+')
for fileName in os.listdir(labelPath):
    currFile = open(labelPath + '/' + fileName, "r")
    lines = currFile.readlines()
    # print("Current File name: ", fileName)
    imgNum = fileName[:6]

    # read image
    img = cv2.imread(imgPath + '/' + imgNum + ".png")
    j = 1 # 'j' counts the number of detections in an image

    # iterating through all the lines in the .txt file
    for currLine in lines:
        #print(truth)
        # if i>3000:
        if i==35000:
            fh.close()
            print('Writing Validation Set')
            label=labels[1]
            fh= open(filePath[label]+'/orientation_labels.txt','w+')
        # if i>20:
        #     continue
        if i%500==0:
            print('Number of Detections:',i)
        values = currLine.split()
        if values[0] != 'Car' and values[0] != 'Van' and values[0] != 'Truck':
            continue
        bbox = values[4:8]
        # print(values[0])
        # generating the crop and writing it into a file
        imgCrop = img[int(float(bbox[1])):int(float(bbox[3])), int(float(bbox[0])):int(float(bbox[2]))]
        # cv2.imshow("Image" + str(i), imgCrop)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # finding observation angle
        alpha = float(values[3])
        orientation=float(values[14])
        fh.write(imgNum+'_' + str(j)+'.png' +' '+ values[0]+' '+values[3]+' '+values[14]+'\n')
        # classifying the detection into label
        # if (0<=alpha<np.pi):
        #     label = 'Positive'
        # else:
        #     label = 'Negative'

        # based upon the label, putting the file into right folder
        cv2.imwrite(filePath[label] + '/' + imgNum + '_' + str(j)+'.png', imgCrop)
        i = i+1
        j = j+1

    currFile.close()
fh.close()
