import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

niiFilePathImg = './nii_files/img'
niiFilePathLabel = './nii_files/label'
outputPath = './output'


for image in os.listdir(niiFilePathImg):
    #Load the scan and extract data using nibabel 
    scan = nib.load(niiFilePathImg + '/' + image)
    scanArray = scan.get_fdata()

    #Examine scan's shape and header 
    scanHeader = scan.header

    #Get and print the scan's shape 
    scanArrayShape = scanArray.shape

    folder_name = image.split('.')[0]

    os.makedirs(outputPath + '/' + folder_name + '/img')

    #Iterate and save scan slices along 2nd dimension
    for i in range(scanArrayShape[2]):
        cv2.imwrite(outputPath + '/' + folder_name + "/img/slice_" + str(i) + '.png', scanArray[:,:,i])

for label in os.listdir(niiFilePathLabel):
    #Load the scan and extract data using nibabel 
    scan = nib.load(niiFilePathLabel + '/' + label)
    scanArray = scan.get_fdata()

    #Examine scan's shape and header 
    scanHeader = scan.header

    #Get and print the scan's shape 
    scanArrayShape = scanArray.shape

    folder_name = label.split('.')[0].replace('label', 'img')

    os.makedirs(outputPath + '/' + folder_name + '/label')

    #Iterate and save scan slices along 2nd dimension
    for i in range(scanArrayShape[2]):
        cv2.imwrite(outputPath + '/' + folder_name + "/label/slice_" + str(i) + '.png', scanArray[:,:,i])


for folder in os.listdir(outputPath):
    outputPathImg = outputPath + '/' + folder + '/img'
    outputPathLabel = outputPath + '/' + folder + '/label'

    for label in os.listdir(outputPathLabel):
        mask = cv2.imread(outputPathLabel + '/' + label, 0)
        
        if np.all(mask == 0):
            os.remove(outputPathLabel + '/' + label)
            os.remove(outputPathImg + '/' + label)
            continue

        unique = np.unique(mask)

        unique = unique[unique != 0]

        for i in range(len(unique)):
            new_img = np.where(mask == unique[i], 255, 0)
            cv2.imwrite(outputPathLabel + '/' + label.split('.')[0] + '_' + str(i) + '.png', new_img)

        os.remove(outputPathLabel + '/' + label)