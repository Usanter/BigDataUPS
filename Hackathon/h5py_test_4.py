#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour générer les images correspondant au deuxième jeu de données
fournies par l'IRT.
"""

import h5py
import numpy as np
from PIL import Image


def assemble_img(filename, start, stop, delay, width):
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
    line_array = np.zeros((16, 16 * delay, 4))
    #Image array init
    img_array = np.zeros((0, 16 * width, 4))
    line_length = delay
    line_nb = 0
    
    for i in range(start, stop + 1):
        patch_array = data["S2"][i,:,:,0:4]
            
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
    empty_line = np.zeros((16, 16 * nb, 4))
    line_array = np.concatenate((line_array, empty_line), axis=1)
    img_array = np.concatenate((img_array, line_array), axis=0)
    return img_array
    
    
"""
BOUNDS:
- Première valeur = premier patch de l'image.
- Deuxième valeur = dernier patch de l'image.
- Troisième valeur = position -1 du premier patch de l'image sur la
                        première ligne.
"""
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
                   [ 9000,   9999, 56]
                   ])

PATH_DATA = 'data/train/eightieth.h5'
OUT_DIR = './images' # Where to store the generated images
width = 16*4 # Width of the image in patches

# Number of images whose bounds are defined
nb_of_images = BOUNDS.shape[0]

for i in range(nb_of_images):
    start = BOUNDS[i, 0]
    stop = BOUNDS[i, 1]
    delay = BOUNDS[i, 2]
    img_array = assemble_img(PATH_DATA, start, stop, delay, width)
    img_array = img_array[:,:,[0, 1, 2]]
    #Normalisation à revoir:
    img = Image.fromarray((img_array * 255/2500).astype(np.uint8), 'RGB')
    img.save(OUT_DIR + '/' + str(i) + '.png')

    