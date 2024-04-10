import cv2
import os
import re
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_bf

import sys


# Get root directory with images
root_image_directory = os.path.abspath('output')
slice_number_regex = r'\d+'

# Lists for mask scores
dataset_click_1 = []
dataset_click_2 = []
dataset_click_5 = []
dataset_click_10 = []
dataset_click_15 = []

# SAM setup
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# Open diretory with individual image set directories - imgXXXX
for image_directory in os.listdir(root_image_directory):
    # Get full path to the directory - path/to/imgXXXX
    image_directory_filepath = os.path.join(root_image_directory, image_directory)
    # Get path to input_image directory - imgXXXX/img
    for dir in os.listdir(image_directory_filepath):
      if dir == 'img':
        input_image_directory = os.path.join(image_directory_filepath, dir)
      elif dir == 'label':
        label_image_directory = os.path.join(image_directory_filepath, dir)


    # Open directory with individual input images - imgXXXX/img/slice_XX
    for input_image in os.listdir(input_image_directory):
        # filepath to individual input images - imgXXXX/img/slice_XX.png
        input_image_filepath = os.path.join(input_image_directory, input_image)

        # Read the input image
        image = cv2.imread(input_image_filepath, cv2.IMREAD_GRAYSCALE)

        # Get slice number
        find_result = re.findall(slice_number_regex, os.path.basename(input_image_filepath))
        if (find_result == []):
          break
        else:
          print(find_result)
          slice_number = find_result[0]

        # Get input masks for current images
        for input_mask in os.listdir( label_image_directory):
            input_label_filepath = os.path.join(label_image_directory, input_mask)

            # Match slice number with correct labels for given input image
            if(re.search(slice_number, input_label_filepath)):
                ground_truth = cv2.imread(input_label_filepath, cv2.IMREAD_GRAYSCALE)
                initial_prediction = cv2.imread(input_label_filepath, cv2.IMREAD_GRAYSCALE)

                prediction_result = sam_predict(image, ground_truth, initial_prediction)
                #sam_predict(image, ground_truth, initial_prediction, predictor)
                dataset_click_1.append(prediction_result[0])
                dataset_click_2.append(prediction_result[1])
                dataset_click_5.append(prediction_result[2])
                dataset_click_10.append(prediction_result[3])
                dataset_click_15.append(prediction_result[4])

            else:
                continue

# Plot graph
plt.figure(figsize = (10,10))
values =  [np.average(dataset_click_1), np.average(dataset_click_2),
           np.average(dataset_click_5), np.average(dataset_click_10),
           np.average(dataset_click_15)]

labels = ['1', '2', '5', '10', '15']
colors = ['green', 'red', 'cyan', 'magenta', 'yellow']
plt.ylim(0, 1)
for i, (label, value) in enumerate(zip(labels, values)):
    plt.bar(i, value, align='center', color=colors[i], label=label, width=0.5)
    plt.text(i, value - 0.04, f'{value:.2f}', ha='center')

plt.xticks(range(len(labels)), labels)
plt.xlabel('Number of clicks')
plt.ylabel('Average IoU')
plt.title('Average IoUs per number of clicks')

# Display the plot
plt.savefig('average_iou.png')

def sam_predict(input_image, ground_truth, prediction):

    # number of clicks
    num_of_clicks = 5

    # click lists
    click_list = []
    type_list = []
    all_masks = []
    all_scores = []

    click_1 = []
    click_2 = []
    click_5 = []
    click_10 = []
    click_15 = []

    masks = []
    scores = []

    # Convert grayscale images so SAM accepts them
    conv_input_image = np.expand_dims(input_image, axis=2)
    conv_input_image = np.repeat(conv_input_image, 3, axis=2)


    # # TODO: Garbage? -> SafariDrip: Garbango
    # # conv_prediction = np.expand_dims(prediction, axis=2)
    # # conv_prediction = np.repeat(conv_prediction, 3, axis=2)
    # # conv_ground_truth = np.expand_dims(ground_truth, axis=2)
    # # conv_ground_truth = np.repeat(conv_ground_truth, 3, axis=2)

    for i in range(num_of_clicks + 1):

        # Generate clicks and append them to the list
        random_pixel_2d_index, click_type = get_random_click(ground_truth, prediction)
        click_list.append([random_pixel_2d_index[1], random_pixel_2d_index[0]])
        type_list.append(click_type)

        click_list_np = np.array(click_list)
        type_list_np = np.array(type_list)

        # Set predictor to our image
        predictor.set_image(conv_input_image)

        # Generate masks, scores and logits
        # for multiple clicks, only best mask input is selected -- OPTIONAL
        if len(type_list) == 1:
            masks, scores, logits = predictor.predict(
                point_coords=click_list_np,
                point_labels=type_list_np,
            )
        else:
            mask_input = logits[np.argmax(scores), :, :]
            masks, scores, logits = predictor.predict(
                point_coords=click_list_np,
                point_labels=type_list_np,
                mask_input=mask_input[None, :, :],
                multimask_output=True
            )

        # Assign new mask
        if i == 0:
          mask_image1 = (masks[0] * 255).astype(np.uint8)  # Convert to uint8 format
          cv2.imwrite('mask_for_pred_assignment1.png', mask_image1)  # Save the mask
          prediction1 = cv2.imread('mask_for_pred_assignment1.png',
                                cv2.IMREAD_GRAYSCALE)  # Load the mask in binary format
          predictions = [prediction1]                        
        else:
          mask_image1 = (masks[0] * 255).astype(np.uint8)  # Convert to uint8 format  
          mask_image2 = (masks[1] * 255).astype(np.uint8)
          mask_image3 = (masks[2] * 255).astype(np.uint8)

          cv2.imwrite('mask_for_pred_assignment1.png', mask_image1)  # Save the mask
          prediction1 = cv2.imread('mask_for_pred_assignment1.png',
                                cv2.IMREAD_GRAYSCALE)  # Load the mask in binary format
          cv2.imwrite('mask_for_pred_assignment2.png', mask_image2)  # Save the mask
          prediction2 = cv2.imread('mask_for_pred_assignment2.png',
                                cv2.IMREAD_GRAYSCALE)  # Load the mask in binary format
          cv2.imwrite('mask_for_pred_assignment3.png', mask_image3)  # Save the mask
          prediction3 = cv2.imread('mask_for_pred_assignment3.png',
                                cv2.IMREAD_GRAYSCALE)  # Load the mask in binary format   
          predictions = [prediction1, prediction2, prediction3]                                     
                                     
        # Calculate IoU for first returned mask
        our_scores = []
        for k in range(len(predictions)):
          our_scores.append(calculate_iou(ground_truth, predictions[k]))

        maximum = max(our_scores)
        index = 0
        for k,j in enumerate(our_scores):
          if j == max:
            index = k

        prediction = predictions[index]
        score = our_scores[index]
        if i == 1:
            click_1.append(score)
        elif i == 2:
            click_2.append(score)
        elif i == 5:
            click_5.append(score)
        elif i == 10:
            click_10.append(score)
        elif i == 15:
            click_15.append(score)

    # Clear the lists for new images
    click_list.clear()
    type_list.clear()

    return [click_1, click_2, click_5, click_10, click_15]

def calculate_iou(ground_truth, prediction):
    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def get_random_click(ground_truth, prediction):
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    prediction = prediction.astype(int)
    gt = ground_truth.astype(int)

    prediction = np.array(prediction)
    ground_truth = np.array(gt)

    D_map = ground_truth - prediction


    D_plus = D_map.copy()
    D_minus = D_map.copy()
    D_plus[D_plus < 0] = 0  # D_plus are false-negative pixels
    D_minus[D_minus > 0] = 0  # D_minus are false-positive pixels
    D_minus = np.abs(D_minus)

    # sum non zero elements of D_minus and D_plus
    sum_D_minus = np.sum(D_minus)
    sum_D_plus = np.sum(D_plus)

    click_type = 0
    if (sum_D_minus > sum_D_plus):
        click_type = 0
        selected_map = D_minus
    else:
        click_type = 1
        selected_map = D_plus

    # get distances of each pixel to the nearest border
    sel_map_transformed = distance_transform_bf(selected_map)
    # make the distances even more significant
    sel_map_exp = np.expm1(sel_map_transformed)
    # change the distances to probabilities of a given pixel being selected
    if np.sum(sel_map_exp) != 0:
        P_map = sel_map_exp / np.sum(sel_map_exp)

        # select a random pixel based on the probabilities
        flattened_probabilities = P_map.flatten()

        random_pixel_index = np.random.choice(np.arange(len(flattened_probabilities)), p=flattened_probabilities)
        random_pixel_2d_index = np.unravel_index(random_pixel_index, P_map.shape)
        return random_pixel_2d_index, click_type
    else:
        pred_white = []
        # iterate over every pixel, if it is white, add it to list
        for i in range(1,prediction.shape[0]):
            for j in range(1,prediction.shape[1]):
                if prediction[i, j] == 1:
                    pred_white.append((i, j))

        # select middle pixel
        random_pixel_2d_index = pred_white[int(len(pred_white)/2)]
        return random_pixel_2d_index, click_type