#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour générer les images correspondant au deuxième jeu de données
fournies par l'IRT.
"""

import h5py
import numpy as np
from PIL import Image
import utils_visu
import cv2


def assemble_img(filename, start, stop, delay, width, colors):
    """
    Generate image array from the patches of a h5 file

    Parameters
    ----------
    filename : String
        Path to the h5 file
    start : int
        Index of the first patch of the image
    stop : int
        Index of the last patch of the image
    delay : int
        Number of empty patches to put on the first line before the image
        starts
    width : int
        Width of the image in patches

    Returns
    -------
    numpy array
        Representation of the image in an array

    """
    data = h5py.File(filename,'r')
    
    #Line array init
    line_array = np.zeros((16, 16 * delay, 7))
    #Image array init
    img_array = np.zeros((0, 16 * width, 7))
    line_length = delay
    line_nb = 0
    
    for i in range(start, stop + 1):
        pixels = data["S2"][i,:,:,0:4]
        label = data['TOP_LANDCOVER'][i][0]
        label_patch = generate_label_patch(label, CLASSES_COLORS)
        patch_array = np.dstack((pixels, label_patch))
            
        if  line_length == width: #The line is complete
            #Concatenate the complete line to the image
            img_array = np.concatenate((img_array, line_array), axis=0)
            #Restart new line
            line_array = patch_array
            line_length = 1
            line_nb += 1
        else: #The line is not complete
            line_length += 1
            #Add the patch to the line
            line_array = np.concatenate((line_array, patch_array), axis=1)
            
    nb = width - line_length
    empty_line = np.zeros((16, 16 * nb, 7))
    line_array = np.concatenate((line_array, empty_line), axis=1)
    img_array = np.concatenate((img_array, line_array), axis=0)
    return img_array
    

def generate_label_patch(label, colors):
    return np.full((16, 16, 3), colors[label])
    
    
CLASSES_COLORS = dict([
(0, [170, 240, 240]),
(1, [255, 255, 100]),
(2, [220, 240, 100]),
(3, [205, 205, 102]),
(4, [0, 100, 0]),
(5, [0, 160, 0]),
(6, [170, 200, 0]),
(7, [0, 60, 0]),
(8, [40, 100, 0]),
(9, [120, 130, 0]),
(10, [140, 160, 0]),
(11, [190, 150, 0]),
(12, [150, 100, 0]),
(13, [255, 180, 50]),
(14, [255, 235, 175]),
(15, [0, 120, 90]),
(16, [0, 150, 120]),
(17, [0, 220, 130]),
(18, [195, 20, 0]),
(19, [255, 245, 215]),
(20, [0, 70, 200]),
(21, [255, 255, 255]),
(22, [0, 0, 0]),
    ])


"""
BOUNDS:
- Première valeur = premier patch de l'image.
- Deuxième valeur = dernier patch de l'image.
- Troisième valeur = position -1 du premier patch de l'image sur la
                        première ligne.
"""
# =============================================================================
# BOUNDS = np.array([[    0,    999, 48],
#                    [ 1000,   1999,  8],
#                    [ 2000,   2999, 56],
#                    [ 3000,   3999, 40],
#                    [ 4000,   4999,  8],
#                    [ 5000,   5279, 40],
#                    [ 5280,   5999,  0],
#                    [ 6000,   6999,  8],
#                    [ 7000,   7863, 32],
#                    [ 7864,   7999,  0],
#                    [ 8000,   8999, 24],
#                    [ 9000,   9999, 56],
#                    [ 10240 ,   10999, 0],   #10586
#                    [ 26000,   26999, 0],  #26819
#                    [ 51000 ,   51999, 8],  #51092
#                    [ 71000,   71999, 8],  #71754
#                    [ 74000,   74999, 16],  #74814
#                    [ 85000,   85999, 0],  #85126
#                    [ 91000,   91999, 48],
#                    [ 168000,   168999, 32],  #169692
#                    [ 183000,   183999, 9],  #183681
#                    [ 190000,   190999, 16],  #169692
#                    [ 207000,   207743, 24],     #207574
#                    ])
# =============================================================================
    

BOUNDS = np.array([[    0,    999, 48],
                   [ 1000,   1999,  8],
                   [ 2000,   2999, 56],
                   [ 3000,   3999, 40],
                   [ 4000,   4999,  8],
                   [ 5000,   5279, 40],
                   [ 5280,   5999,  0],
                   [ 6000,   6999,  8],
                   [ 7000,   7863, 32],
                   [ 7864,   7999,  0],
                   [ 8000,   8999, 24],
                   [ 9000,   9999, 56],
                   [ 10240 ,   10999, 0],   #10586
                   [ 26000,   26999, 0],  #26819
                   [ 51000 ,   51999, 8],  #51092
                   [ 71000,   71999, 8],  #71754
                   [ 74000,   74999, 16],  #74814
                   [ 85000,   85999, 0],  #85126
                   [ 91000,   91999, 48],
                   [ 168000,   168999, 32],  #169692
                   [ 183000,   183999, 9],  #183681
                   [ 190000,   190999, 16],  #169692
                   [ 207000,   207743, 24],     #207574
                   ])
    
    

#PATH_DATA = 'data/train/eightieth.h5'
PATH_DATA = 'data/pred_teachers/pred_eighties_from_half_1.h5'
OUT_DIR = './images_test' # Where to store the generated images
width = 16*4 # Width of the image in patches

# Number of images whose bounds are defined
nb_of_images = BOUNDS.shape[0]

# get the histogram equalization table for RGB data
data = h5py.File(PATH_DATA,'r')
cdf_rgb = utils_visu.get_equalization_table(data["S2"][:,:,:,0:3], type = 'flat')

# get the histogram equalization table for infrared data
cdf_ir = utils_visu.get_equalization_table(data["S2"][:,:,:,3])

for i in range(nb_of_images):
    start = BOUNDS[i, 0]
    stop = BOUNDS[i, 1]
    delay = BOUNDS[i, 2]
    array = assemble_img(PATH_DATA, start, stop, delay, width, CLASSES_COLORS)
    
    # RGB
    rgb = array[:,:,[0, 1, 2]]
    #Histogram equalization
    rgb = utils_visu.equalize_patch_hist(rgb, cdf_rgb)
    
    # IR
    ir_1D = array[:,:,3]
    ir_1D = utils_visu.equalize_patch_hist(ir_1D, cdf_ir)
    ir = cv2.cvtColor(ir_1D, cv2.COLOR_GRAY2RGB)
    
    #Label
    label = array[:,:,[4, 5, 6]]
    
    #Mix
    mix = np.multiply(ir/255, label)
    
    
    array_tot = np.concatenate((mix, rgb, ir,label), axis=0)
    
    img = Image.fromarray(array_tot.astype(np.uint8), 'RGB')
    img.save(OUT_DIR + '/' + str(i) + '.png')

    