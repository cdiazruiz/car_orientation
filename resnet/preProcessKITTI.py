# just importing stuff
import os
import sys
import cv2
import csv
import numpy as np
import matplotlib
import matplotlib.image as mpimg
from skimage import io



# Main code here:

# Just 10 data files for reading
# labelPath = '/media/vikram/DATA/ASL/VehicleOrientation/Code/labels'
# imgPath = '/media/vikram/DATA/ASL/VehicleOrientation/Code/images'

# File paths for writing
cropLbPath = '/home/cad297/Documents/ResNet18/Kitti/postLabels'
cropImPath = '/home/cad297/Documents/ResNet18/Kitti/postImages'

# Entire data files
labelPath = '/home/cad297/Documents/ResNet18/Kitti/training/label_2'
imgPath = '/home/cad297/Documents/ResNet18/classification_models-master/orientation_data/Train/
i=1
imgNames= [s for s in os.listdir(imgPath)  if "png" in s] #checking whether filenames are png
for fileName in imgNames:
    if filename[-4:]
    img = cv2.imread(img+fileName)
    j = 1
    currWriteFile = open(cropLbPath + '/' + imgNum + '.txt', "w")

    # iterating through all the lines in the .txt file
    for currLine in lines:
        #print(truth)
        if i>10:
            continue
        values = currLine.split()
        if values[0] != 'Car' and values[0] != 'Van' and values[0] != 'Truck':
            continue
        bbox = values[4:8]
        print(values[0])
        # generating the crop and writing it into a file
        imgCrop = img[int(float(bbox[1])):int(float(bbox[3])), int(float(bbox[0])):int(float(bbox[2]))]
        cv2.imshow("Image" + str(i), imgCrop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite(cropImPath + '/' + imgNum + '_' + str(j)+'.png', imgCrop)

        # Writing the orientation into the file
        currWriteFile.write(values[0] + ' ') # label
        currWriteFile.write(values[3] + ' ') # observation angle
        currWriteFile.write(values[14]) # orientation
        currWriteFile.write('\n')
        i = i+1
        j = j+1

    # if no content in the file i.e. j==1, then delete the file
    if j == 1:
        os.remove(cropLbPath + '/' + imgNum + '.txt')

    currFile.close()
