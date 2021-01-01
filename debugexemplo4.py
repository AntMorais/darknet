import argparse
import os
import glob
import random
import time
import cv2
import numpy as np
import darknet
import darknet_images

#!python darknet_images.py --input /content/drive/MyDrive/FIRELOC_DATA/train.txt --batch_size 1 --weights \
#/content/drive/MyDrive/FIRELOC_DATA/yolov4.weights --dont_show --ext_output --config_file cfg/yolov4.cfg \
#--data_file cfg/coco.data




# Parâmetros todos devem ser globais ou estarem numa função que encapsula o explain_instance()
# Input é um numpy array de imagens onde cada imagem é um ndarray de shape (channel, height, width)
# Output é um numpy array com shape (image index, classes) com as probabilidades de ser cada classe
#def predict_para_lime(input):




# so funciona para uma imagem
def funcao_classificacao_lime(image_numpy):

    probabilities = np.zeros((len(image_numpy), len(class_names)))
    for i in range(len(image_numpy)):
        img = image_numpy[i]
        image, detections = darknet_images.image_detection_lime(
                img, network, class_names,
                class_colors, thresh
                )

        if save_labels:
            darknet_images.save_annotations(image_name, image, detections, class_names)
        #darknet.print_detections(detections, ext_output)
        if not dont_show:
            cv2.imshow('Inference', image)
            if cv2.waitKey() & 0xFF == ord('q'):
                return

        with open("probabilityArray.txt","r") as prob_array:
            lines = [float(line.rstrip()) for line in prob_array]
            probabilities[i] = lines
    return probabilities





txt_input = "../FIRELOC_DATA/train.txt"
batch_size = True
weights = "../FIRELOC_DATA/yolov4.weights"
dont_show = True
ext_output = True
save_labels = True
config_file = "cfg/yolov4.cfg"
data_file = "cfg/coco.data"
thresh = 0.25

images = darknet_images.load_images(txt_input)
image_name = images[0]
network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
        batch_size
    )

def get_most_confident_bbox():
    # First run through Yolo to check if
    image_original, detections = darknet_images.image_detection(
                image_name, network, class_names,
                class_colors, thresh
                )
    darknet.print_detections(detections, args.ext_output)
    # take the detection that had the most confidence and write the coordinates to a txt file
    # we divide the coordinates by the dimensions because we need relative coordinates
    coordinates = detections[-1][2]
    with open("coordinates.txt","w+") as coord_file:
        coordinates = darknet_images.convert2relative(image_original,coordinates)
        for _coord in coordinates:
            coord_file.write(str(_coord)+"\n")


get_most_confident_bbox()



width = darknet.network_width(network)
height = darknet.network_height(network)
image = cv2.imread(image_name)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image_rgb, (width, height),
                            interpolation=cv2.INTER_LINEAR)



from lime import lime_image
#Objeto do tipo LimeImageExplainer
explainer = lime_image.LimeImageExplainer()
#Objeto do tipo ImageExplanation
explanation = explainer.explain_instance(np.array(image_resized), 
                                         funcao_classificacao_lime, # classification function
                                         top_labels=5, 
                                         hide_color=0, 
                                         num_samples=5000) # number of images that will be sent to classification function


import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
img_boundry2 = mark_boundaries(temp/255.0, mask)
plt.imshow(img_boundry2)
plt.show()
